/**
 * MatrixFlow — Interactive Matrix Multiplication Engine & Visualizer
 * Core Application Logic
 */

(function () {
  'use strict';

  // --- Audio Feedback Engine (Web Audio API) ---
  class SoundEffects {
    constructor() {
      this.enabled = true;
      this.ctx = null;
    }

    init() {
      if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
      }
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
    }

    playStepSound(kIndex = 0, maxK = 1) {
      if (!this.enabled) return;
      try {
        this.init();
        if (!this.ctx) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        const baseFreq = 480;
        const freq = baseFreq + (kIndex / Math.max(1, maxK)) * 240;

        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        
        gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.08);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start();
        osc.stop(this.ctx.currentTime + 0.08);
      } catch (e) {
        // Audio policy or unsupported
      }
    }

    playCompletionChime() {
      if (!this.enabled) return;
      try {
        this.init();
        if (!this.ctx) return;
        const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
        notes.forEach((freq, idx) => {
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          const startTime = this.ctx.currentTime + idx * 0.07;
          
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, startTime);

          gain.gain.setValueAtTime(0.05, startTime);
          gain.gain.exponentialRampToValueAtTime(0.0001, startTime + 0.28);

          osc.connect(gain);
          gain.connect(this.ctx.destination);

          osc.start(startTime);
          osc.stop(startTime + 0.28);
        });
      } catch (e) {
        // Audio error silent
      }
    }
  }

  const sfx = new SoundEffects();

  // --- Application State ---
  const state = {
    matrixCount: 2, // 2 or 3
    rowsA: 2,
    colsA: 3,
    rowsB: 3,
    colsB: 2,
    rowsC: 2,
    colsC: 2,
    isLinked: true,
    matrixA: [],
    matrixB: [],
    matrixC: [],        // Result in 2-matrix mode, or reference to final result
    matrixCInput: [],   // 3rd input matrix in 3-matrix mode
    intermediateAB: [], // Intermediate product (A x B) in 3-matrix mode
    matrixR: [],        // Final result (AB x C) in 3-matrix mode
    
    // Playback and Stepper state
    steps: [],
    currentStepIndex: -1,
    isPlaying: false,
    playTimer: null,
    playSpeed: 1.0,
    stepMode: 'subterm', // 'subterm' or 'cell'
    selectedCellC: null, // { i, j }
    activeModal: null
  };

  // --- Matrix Math Helpers ---
  function createMatrix(rows, cols, fillFn = () => 0) {
    const mat = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        row.push(fillFn(r, c));
      }
      mat.push(row);
    }
    return mat;
  }

  function randomInt(min = -9, max = 9) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function canMultiply() {
    if (state.matrixCount === 2) {
      return state.colsA === state.rowsB;
    } else {
      return (state.colsA === state.rowsB) && (state.colsB === state.rowsC);
    }
  }

  function computeResultMatrix() {
    if (!canMultiply()) {
      state.matrixC = [];
      state.intermediateAB = [];
      state.matrixR = [];
      return;
    }

    if (state.matrixCount === 2) {
      const m = state.rowsA;
      const kDim = state.colsA;
      const n = state.colsB;
      const C = [];

      for (let i = 0; i < m; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let k = 0; k < kDim; k++) {
            const valA = Number(state.matrixA[i]?.[k]) || 0;
            const valB = Number(state.matrixB[k]?.[j]) || 0;
            sum += valA * valB;
          }
          row.push(sum);
        }
        C.push(row);
      }
      state.matrixC = C;
    } else {
      // 3-matrix mode:
      // Stage 1: AB = A (m x k) * B (k x l) -> shape (m x l)
      const m = state.rowsA;
      const kDim = state.colsA;
      const lDim = state.colsB;
      const n = state.colsC;

      const AB = [];
      for (let i = 0; i < m; i++) {
        const row = [];
        for (let p = 0; p < lDim; p++) {
          let sum = 0;
          for (let k = 0; k < kDim; k++) {
            const valA = Number(state.matrixA[i]?.[k]) || 0;
            const valB = Number(state.matrixB[k]?.[p]) || 0;
            sum += valA * valB;
          }
          row.push(sum);
        }
        AB.push(row);
      }
      state.intermediateAB = AB;

      // Stage 2: R = AB (m x l) * C (l x n) -> shape (m x n)
      const R = [];
      for (let i = 0; i < m; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let p = 0; p < lDim; p++) {
            const valAB = Number(AB[i]?.[p]) || 0;
            const valC = Number(state.matrixCInput[p]?.[j]) || 0;
            sum += valAB * valC;
          }
          row.push(sum);
        }
        R.push(row);
      }
      state.matrixR = R;
      state.matrixC = R; // Alias for unified inspection
    }
  }

  /**
   * Generates flattened sequence of computation steps.
   */
  function generateSteps() {
    if (!canMultiply()) {
      state.steps = [];
      return;
    }

    const steps = [];

    if (state.matrixCount === 2) {
      const m = state.rowsA;
      const kDim = state.colsA;
      const n = state.colsB;

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let runningSum = 0;
          const cellTerms = [];

          for (let k = 0; k < kDim; k++) {
            const valA = Number(state.matrixA[i]?.[k]) || 0;
            const valB = Number(state.matrixB[k]?.[j]) || 0;
            const prod = valA * valB;
            runningSum += prod;

            const term = {
              k,
              valA,
              valB,
              prod,
              runningSum
            };
            cellTerms.push(term);

            steps.push({
              stage: 1,
              type: 'subterm',
              i,
              j,
              k,
              valA,
              valB,
              prod,
              runningSum,
              isCellEnd: k === kDim - 1,
              cellTerms: [...cellTerms],
              cellTotalTerms: kDim,
              targetLabel: `C[${i + 1}, ${j + 1}]`
            });
          }
        }
      }
    } else {
      // 3-matrix mode:
      // Stage 1: A x B -> Intermediate AB
      const m = state.rowsA;
      const kDim = state.colsA;
      const lDim = state.colsB;
      const n = state.colsC;

      for (let i = 0; i < m; i++) {
        for (let p = 0; p < lDim; p++) {
          let runningSum = 0;
          const cellTerms = [];

          for (let k = 0; k < kDim; k++) {
            const valA = Number(state.matrixA[i]?.[k]) || 0;
            const valB = Number(state.matrixB[k]?.[p]) || 0;
            const prod = valA * valB;
            runningSum += prod;
            cellTerms.push({ k, valA, valB, prod, runningSum });

            steps.push({
              stage: 1,
              type: 'subterm',
              i,
              j: p, // col in intermediate
              k,
              valA,
              valB,
              prod,
              runningSum,
              isCellEnd: k === kDim - 1,
              cellTerms: [...cellTerms],
              cellTotalTerms: kDim,
              targetLabel: `(A×B)[${i + 1}, ${p + 1}]`
            });
          }
        }
      }

      // Stage 2: AB x C -> Result R
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let runningSum = 0;
          const cellTerms = [];

          for (let p = 0; p < lDim; p++) {
            const valAB = Number(state.intermediateAB[i]?.[p]) || 0;
            const valC = Number(state.matrixCInput[p]?.[j]) || 0;
            const prod = valAB * valC;
            runningSum += prod;
            cellTerms.push({ k: p, valA: valAB, valB: valC, prod, runningSum });

            steps.push({
              stage: 2,
              type: 'subterm',
              i,
              j,
              k: p,
              valA: valAB,
              valB: valC,
              prod,
              runningSum,
              isCellEnd: p === lDim - 1,
              cellTerms: [...cellTerms],
              cellTotalTerms: lDim,
              targetLabel: `R[${i + 1}, ${j + 1}]`
            });
          }
        }
      }
    }

    state.steps = steps;
  }

  // --- UI Elements Cache ---
  const el = {
    // Mode Switcher & Swap
    mode2MatricesBtn: document.getElementById('mode2MatricesBtn'),
    mode3MatricesBtn: document.getElementById('mode3MatricesBtn'),
    swapABTopBtn: document.getElementById('swapABTopBtn'),
    swapStageBtn: document.getElementById('swapStageBtn'),

    // Inputs Matrix A & B
    rowsA: document.getElementById('rowsA'),
    colsA: document.getElementById('colsA'),
    rowsB: document.getElementById('rowsB'),
    colsB: document.getElementById('colsB'),
    decRowsA: document.getElementById('decRowsA'),
    incRowsA: document.getElementById('incRowsA'),
    decColsA: document.getElementById('decColsA'),
    incColsA: document.getElementById('incColsA'),
    decRowsB: document.getElementById('decRowsB'),
    incRowsB: document.getElementById('incRowsB'),
    decColsB: document.getElementById('decColsB'),
    incColsB: document.getElementById('incColsB'),
    labelColsB: document.getElementById('labelColsB'),

    // Inputs Matrix C (3-Matrix Mode)
    dimBoxMatrixC: document.getElementById('dimBoxMatrixC'),
    rowsC: document.getElementById('rowsC'),
    colsC: document.getElementById('colsC'),
    decRowsC: document.getElementById('decRowsC'),
    incRowsC: document.getElementById('incRowsC'),
    decColsC: document.getElementById('decColsC'),
    incColsC: document.getElementById('incColsC'),
    cardMatrixCInput: document.getElementById('cardMatrixCInput'),
    gridCInput: document.getElementById('gridCInput'),
    badgeCInput: document.getElementById('badgeCInput'),
    emptyCBtn: document.getElementById('emptyCBtn'),
    randCBtn: document.getElementById('randCBtn'),
    clearCBtn: document.getElementById('clearCBtn'),
    opTimesBC: document.getElementById('opTimesBC'),

    // Intermediate Product Card (A x B)
    cardIntermediate: document.getElementById('cardIntermediate'),
    gridIntermediate: document.getElementById('gridIntermediate'),
    badgeIntermediate: document.getElementById('badgeIntermediate'),
    
    // Status & Badges
    dimLinkIndicator: document.getElementById('dimLinkIndicator'),
    dimStatusBadge: document.getElementById('dimStatusBadge'),
    syncDimBtn: document.getElementById('syncDimBtn'),
    innerKVal: document.getElementById('innerKVal'),
    resDimText: document.getElementById('resDimText'),
    resDimSub: document.getElementById('resDimSub'),
    resLabelBadge: document.getElementById('resLabelBadge'),
    resCardTitle: document.getElementById('resCardTitle'),
    resCardHint: document.getElementById('resCardHint'),
    dimensionAlert: document.getElementById('dimensionAlert'),
    dimensionAlertText: document.getElementById('dimensionAlertText'),
    fixDimensionBtn: document.getElementById('fixDimensionBtn'),

    cardMatrixA: document.getElementById('cardMatrixA'),
    cardMatrixB: document.getElementById('cardMatrixB'),
    cardMatrixC: document.getElementById('cardMatrixC'),

    badgeA: document.getElementById('badgeA'),
    badgeB: document.getElementById('badgeB'),
    badgeC: document.getElementById('badgeC'),

    // Grids
    gridA: document.getElementById('gridA'),
    gridB: document.getElementById('gridB'),
    gridC: document.getElementById('gridC'),

    // Player Toolbar
    btnFirst: document.getElementById('btnFirst'),
    btnPrev: document.getElementById('btnPrev'),
    btnPlayPause: document.getElementById('btnPlayPause'),
    playPauseText: document.getElementById('playPauseText'),
    iconPlay: document.querySelector('.icon-play'),
    iconPause: document.querySelector('.icon-pause'),
    btnNext: document.getElementById('btnNext'),
    btnLast: document.getElementById('btnLast'),
    btnReset: document.getElementById('btnReset'),
    modeSubTerm: document.getElementById('modeSubTerm'),
    modeCell: document.getElementById('modeCell'),
    speedRange: document.getElementById('speedRange'),
    speedValueText: document.getElementById('speedValueText'),

    // Progress Bar
    stepCounterText: document.getElementById('stepCounterText'),
    stepTargetText: document.getElementById('stepTargetText'),
    stepPercentText: document.getElementById('stepPercentText'),
    progressBarFill: document.getElementById('progressBarFill'),

    // Inspector Deck
    currentCellName: document.getElementById('currentCellName'),
    ruleRow: document.getElementById('ruleRow'),
    ruleCol: document.getElementById('ruleCol'),
    termsBreakdownRow: document.getElementById('termsBreakdownRow'),
    formulaBody: document.getElementById('formulaBody'),
    runningTotalValue: document.getElementById('runningTotalValue'),
    narrativeExplanation: document.getElementById('narrativeExplanation'),

    // Header & Actions
    soundToggleBtn: document.getElementById('soundToggleBtn'),
    soundOnIcon: document.querySelector('.sound-on-icon'),
    soundOffIcon: document.querySelector('.sound-off-icon'),
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeSunIcon: document.querySelector('.theme-icon-sun'),
    themeMoonIcon: document.querySelector('.theme-icon-moon'),
    presetBtn: document.getElementById('presetBtn'),
    presetMenu: document.getElementById('presetMenu'),
    randomizeAllBtn: document.getElementById('randomizeAllBtn'),
    clearAllBtn: document.getElementById('clearAllBtn'),
    identityBtn: document.getElementById('identityBtn'),
    randABtn: document.getElementById('randABtn'),
    clearABtn: document.getElementById('clearABtn'),
    emptyABtn: document.getElementById('emptyABtn'),
    randBBtn: document.getElementById('randBBtn'),
    clearBBtn: document.getElementById('clearBBtn'),
    emptyBBtn: document.getElementById('emptyBBtn'),
    emptyAllBtn: document.getElementById('emptyAllBtn'),
    calculateBtn: document.getElementById('calculateBtn'),
    pasteMatrixBtn: document.getElementById('pasteMatrixBtn'),
    exportBtn: document.getElementById('exportBtn'),

    // Modals
    pasteModal: document.getElementById('pasteModal'),
    closePasteModalBtn: document.getElementById('closePasteModalBtn'),
    cancelPasteBtn: document.getElementById('cancelPasteBtn'),
    applyPasteBtn: document.getElementById('applyPasteBtn'),
    matrixRawInput: document.getElementById('matrixRawInput'),
    tabMatrixA: document.getElementById('tabMatrixA'),
    tabMatrixB: document.getElementById('tabMatrixB'),
    tabMatrixCInput: document.getElementById('tabMatrixCInput'),
    exampleIdentity: document.getElementById('exampleIdentity'),
    exampleRotation: document.getElementById('exampleRotation'),
    exampleFibonacci: document.getElementById('exampleFibonacci'),

    exportModal: document.getElementById('exportModal'),
    closeExportModalBtn: document.getElementById('closeExportModalBtn'),
    closeExportBtn: document.getElementById('closeExportBtn'),
    copyExportBtn: document.getElementById('copyExportBtn'),
    exportContentArea: document.getElementById('exportContentArea'),
    copyStatusText: document.getElementById('copyStatusText'),
    tabExportLatex: document.getElementById('tabExportLatex'),
    tabExportJSON: document.getElementById('tabExportJSON'),
    tabExportCSV: document.getElementById('tabExportCSV'),
    tabExportPython: document.getElementById('tabExportPython')
  };

  let activePasteTarget = 'A';
  let activeExportFormat = 'latex';

  // --- Initial Setup & Seed Data ---
  function initDefaultMatrices() {
    state.matrixA = [
      [2, -1, 3],
      [4, 0, 5]
    ];
    state.matrixB = [
      [1, 4],
      [3, 2],
      [-2, 1]
    ];
    state.rowsA = 2;
    state.colsA = 3;
    state.rowsB = 3;
    state.colsB = 2;

    // Default Matrix C for 3-Matrix mode
    state.matrixCInput = [
      [1, -2],
      [2, 3]
    ];
    state.rowsC = 2;
    state.colsC = 2;

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1); // start in idle ready state
  }

  // --- Matrix Rendering ---
  function renderAll() {
    syncDimensionUI();
    renderGridA();
    renderGridB();
    if (state.matrixCount === 3) {
      renderGridCInput();
      renderGridIntermediate();
    }
    renderGridC();
    updateInspectorUI();
  }

  function syncDimensionUI() {
    el.rowsA.value = state.rowsA;
    el.colsA.value = state.colsA;
    el.rowsB.value = state.rowsB;
    el.colsB.value = state.colsB;
    if (state.matrixCount === 3 && el.rowsC && el.colsC) {
      el.rowsC.value = state.rowsC;
      el.colsC.value = state.colsC;
    }

    el.badgeA.textContent = `${state.rowsA} × ${state.colsA}`;
    el.badgeB.textContent = `${state.rowsB} × ${state.colsB}`;
    if (state.matrixCount === 3 && el.badgeCInput && el.badgeIntermediate) {
      el.badgeCInput.textContent = `${state.rowsC} × ${state.colsC}`;
      el.badgeIntermediate.textContent = `${state.rowsA} × ${state.colsB}`;
    }
    el.innerKVal.textContent = state.colsA;

    const valid = canMultiply();
    if (valid) {
      const outRows = state.rowsA;
      const outCols = state.matrixCount === 2 ? state.colsB : state.colsC;
      el.resDimText.textContent = `${outRows} × ${outCols}`;
      el.badgeC.textContent = `${outRows} × ${outCols}`;
      el.dimStatusBadge.classList.remove('mismatch');
      el.dimStatusBadge.querySelector('.icon-match').classList.remove('hidden');
      el.dimStatusBadge.querySelector('.icon-error').classList.add('hidden');
      el.dimensionAlert.classList.add('hidden');
      el.btnPlayPause.disabled = false;
      el.btnNext.disabled = false;
      el.btnPrev.disabled = false;
    } else {
      el.resDimText.textContent = 'Invalid';
      el.badgeC.textContent = 'Incompatible';
      el.dimStatusBadge.classList.add('mismatch');
      el.dimStatusBadge.querySelector('.icon-match').classList.add('hidden');
      el.dimStatusBadge.querySelector('.icon-error').classList.remove('hidden');
      const reason = state.matrixCount === 2
        ? `Columns of A (${state.colsA}) must equal Rows of B (${state.rowsB}).`
        : `Cols A (${state.colsA}) must equal Rows B (${state.rowsB}), AND Cols B (${state.colsB}) must equal Rows C (${state.rowsC}).`;
      el.dimensionAlertText.textContent = `Dimension mismatch: ${reason}`;
      el.dimensionAlert.classList.remove('hidden');
      stopPlaying();
      el.btnPlayPause.disabled = true;
      el.btnNext.disabled = true;
      el.btnPrev.disabled = true;
    }
  }

  // Helper to dynamically toggle zero, non-zero, and empty styling
  function updateCellZeroState(cellElem, value) {
    if (!cellElem) return;
    const isEmpty = (value === '' || value === null || value === undefined || (typeof value === 'number' && isNaN(value)));
    cellElem.classList.toggle('cell-empty', isEmpty);
    if (isEmpty) {
      cellElem.classList.remove('cell-zero', 'cell-nonzero');
      return;
    }
    const num = Number(value);
    const isZero = (num === 0 || Object.is(num, -0));
    cellElem.classList.toggle('cell-zero', isZero);
    cellElem.classList.toggle('cell-nonzero', !isZero);
  }

  function renderGridA() {
    el.gridA.innerHTML = '';
    el.gridA.style.gridTemplateRows = `repeat(${state.rowsA}, 52px)`;
    el.gridA.style.gridTemplateColumns = `repeat(${state.colsA}, 58px)`;

    for (let r = 0; r < state.rowsA; r++) {
      for (let c = 0; c < state.colsA; c++) {
        const val = state.matrixA[r]?.[c];
        const isEmpty = (val === '' || val === null || val === undefined);
        const cell = document.createElement('div');
        cell.className = 'matrix-cell cell-a';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.id = `cell-A-${r}-${c}`;
        updateCellZeroState(cell, val);

        const indexBadge = document.createElement('span');
        indexBadge.className = 'cell-index-sub';
        indexBadge.textContent = `${r + 1},${c + 1}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'cell-input';
        input.value = isEmpty ? '' : val;
        input.placeholder = '·';
        input.dataset.row = r;
        input.dataset.col = c;

        // Instant shade and empty feedback on keystroke
        input.addEventListener('input', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixA[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const liveVal = parseFloat(raw);
            if (isNaN(liveVal)) {
              state.matrixA[r][c] = '';
              updateCellZeroState(cell, '');
            } else {
              state.matrixA[r][c] = liveVal;
              updateCellZeroState(cell, liveVal);
            }
          }
        });

        input.addEventListener('change', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixA[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const parsedVal = parseFloat(raw);
            state.matrixA[r][c] = isNaN(parsedVal) ? '' : parsedVal;
            updateCellZeroState(cell, state.matrixA[r][c]);
          }
          onMatrixDataChanged();
        });

        input.addEventListener('keydown', handleCellKeyNav);

        cell.appendChild(indexBadge);
        cell.appendChild(input);
        el.gridA.appendChild(cell);
      }
    }
  }

  function renderGridB() {
    el.gridB.innerHTML = '';
    el.gridB.style.gridTemplateRows = `repeat(${state.rowsB}, 52px)`;
    el.gridB.style.gridTemplateColumns = `repeat(${state.colsB}, 58px)`;

    for (let r = 0; r < state.rowsB; r++) {
      for (let c = 0; c < state.colsB; c++) {
        const val = state.matrixB[r]?.[c];
        const isEmpty = (val === '' || val === null || val === undefined);
        const cell = document.createElement('div');
        cell.className = 'matrix-cell cell-b';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.id = `cell-B-${r}-${c}`;
        updateCellZeroState(cell, val);

        const indexBadge = document.createElement('span');
        indexBadge.className = 'cell-index-sub';
        indexBadge.textContent = `${r + 1},${c + 1}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'cell-input';
        input.value = isEmpty ? '' : val;
        input.placeholder = '·';
        input.dataset.row = r;
        input.dataset.col = c;

        // Instant shade and empty feedback on keystroke
        input.addEventListener('input', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixB[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const liveVal = parseFloat(raw);
            if (isNaN(liveVal)) {
              state.matrixB[r][c] = '';
              updateCellZeroState(cell, '');
            } else {
              state.matrixB[r][c] = liveVal;
              updateCellZeroState(cell, liveVal);
            }
          }
        });

        input.addEventListener('change', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixB[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const parsedVal = parseFloat(raw);
            state.matrixB[r][c] = isNaN(parsedVal) ? '' : parsedVal;
            updateCellZeroState(cell, state.matrixB[r][c]);
          }
          onMatrixDataChanged();
        });

        input.addEventListener('keydown', handleCellKeyNav);

        cell.appendChild(indexBadge);
        cell.appendChild(input);
        el.gridB.appendChild(cell);
      }
    }
  }

  function renderGridCInput() {
    if (!el.gridCInput) return;
    el.gridCInput.innerHTML = '';
    el.gridCInput.style.gridTemplateRows = `repeat(${state.rowsC}, 52px)`;
    el.gridCInput.style.gridTemplateColumns = `repeat(${state.colsC}, 58px)`;

    for (let r = 0; r < state.rowsC; r++) {
      for (let c = 0; c < state.colsC; c++) {
        const val = state.matrixCInput[r]?.[c];
        const isEmpty = (val === '' || val === null || val === undefined);
        const cell = document.createElement('div');
        cell.className = 'matrix-cell cell-c-input';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.id = `cell-CIn-${r}-${c}`;
        updateCellZeroState(cell, val);

        const indexBadge = document.createElement('span');
        indexBadge.className = 'cell-index-sub';
        indexBadge.textContent = `${r + 1},${c + 1}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'cell-input';
        input.value = isEmpty ? '' : val;
        input.placeholder = '·';
        input.dataset.row = r;
        input.dataset.col = c;

        input.addEventListener('input', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixCInput[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const liveVal = parseFloat(raw);
            if (isNaN(liveVal)) {
              state.matrixCInput[r][c] = '';
              updateCellZeroState(cell, '');
            } else {
              state.matrixCInput[r][c] = liveVal;
              updateCellZeroState(cell, liveVal);
            }
          }
        });

        input.addEventListener('change', (e) => {
          const raw = e.target.value.trim();
          if (raw === '') {
            state.matrixCInput[r][c] = '';
            updateCellZeroState(cell, '');
          } else {
            const parsedVal = parseFloat(raw);
            state.matrixCInput[r][c] = isNaN(parsedVal) ? '' : parsedVal;
            updateCellZeroState(cell, state.matrixCInput[r][c]);
          }
          onMatrixDataChanged();
        });

        input.addEventListener('keydown', handleCellKeyNav);

        cell.appendChild(indexBadge);
        cell.appendChild(input);
        el.gridCInput.appendChild(cell);
      }
    }
  }

  function renderGridIntermediate() {
    if (!el.gridIntermediate) return;
    el.gridIntermediate.innerHTML = '';
    const m = state.rowsA;
    const lDim = state.colsB;
    el.gridIntermediate.style.gridTemplateRows = `repeat(${m}, 52px)`;
    el.gridIntermediate.style.gridTemplateColumns = `repeat(${lDim}, 58px)`;

    for (let r = 0; r < m; r++) {
      for (let c = 0; c < lDim; c++) {
        const val = state.intermediateAB[r]?.[c] ?? 0;
        const cell = document.createElement('div');
        cell.className = 'matrix-cell cell-intermediate';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.id = `cell-Inter-${r}-${c}`;
        cell.title = `Intermediate Product (A × B)[${r + 1}, ${c + 1}]`;
        updateCellZeroState(cell, val);

        const indexBadge = document.createElement('span');
        indexBadge.className = 'cell-index-sub';
        indexBadge.textContent = `${r + 1},${c + 1}`;

        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'cell-value-display';
        valueDisplay.id = `val-Inter-${r}-${c}`;
        valueDisplay.textContent = val;

        cell.appendChild(indexBadge);
        cell.appendChild(valueDisplay);
        el.gridIntermediate.appendChild(cell);
      }
    }
  }

  function renderGridC() {
    el.gridC.innerHTML = '';
    if (!canMultiply()) return;

    const outRows = state.rowsA;
    const outCols = state.matrixCount === 2 ? state.colsB : state.colsC;

    el.gridC.style.gridTemplateRows = `repeat(${outRows}, 52px)`;
    el.gridC.style.gridTemplateColumns = `repeat(${outCols}, 58px)`;

    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        const val = state.matrixC[r]?.[c] ?? 0;
        const cell = document.createElement('div');
        cell.className = 'matrix-cell cell-c';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.id = `cell-C-${r}-${c}`;
        cell.title = `Click to inspect computation of Result[${r + 1}, ${c + 1}]`;
        updateCellZeroState(cell, val);

        const indexBadge = document.createElement('span');
        indexBadge.className = 'cell-index-sub';
        indexBadge.textContent = `${r + 1},${c + 1}`;

        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'cell-value-display';
        valueDisplay.id = `val-C-${r}-${c}`;
        valueDisplay.textContent = val;

        cell.appendChild(indexBadge);
        cell.appendChild(valueDisplay);

        cell.addEventListener('click', () => {
          jumpToCell(r, c);
        });

        el.gridC.appendChild(cell);
      }
    }
  }

  // --- Keyboard navigation inside matrices ---
  function handleCellKeyNav(e) {
    const isArrow = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key);
    if (!isArrow) return;

    const row = parseInt(e.target.dataset.row, 10);
    const col = parseInt(e.target.dataset.col, 10);
    const isA = e.target.closest('#gridA') !== null;
    const isB = e.target.closest('#gridB') !== null;
    const maxR = isA ? state.rowsA : (isB ? state.rowsB : state.rowsC);
    const maxC = isA ? state.colsA : (isB ? state.colsB : state.colsC);
    const prefix = isA ? 'A' : (isB ? 'B' : 'CIn');

    let nextR = row;
    let nextC = col;

    if (e.key === 'ArrowUp') nextR = Math.max(0, row - 1);
    if (e.key === 'ArrowDown') nextR = Math.min(maxR - 1, row + 1);
    if (e.key === 'ArrowLeft') nextC = Math.max(0, col - 1);
    if (e.key === 'ArrowRight') nextC = Math.min(maxC - 1, col + 1);

    if (nextR !== row || nextC !== col) {
      e.preventDefault();
      const targetInput = document.querySelector(`#cell-${prefix}-${nextR}-${nextC} input`);
      if (targetInput) {
        targetInput.focus();
        targetInput.select();
      }
    }
  }

  function onMatrixDataChanged() {
    computeResultMatrix();
    generateSteps();

    // Refresh intermediate grid if 3-matrix mode
    if (state.matrixCount === 3) {
      for (let r = 0; r < state.rowsA; r++) {
        for (let c = 0; c < state.colsB; c++) {
          const valElem = document.getElementById(`val-Inter-${r}-${c}`);
          const cellInter = document.getElementById(`cell-Inter-${r}-${c}`);
          const val = state.intermediateAB[r]?.[c] ?? 0;
          if (valElem) valElem.textContent = val;
          if (cellInter) updateCellZeroState(cellInter, val);
        }
      }
    }

    // Refresh result grid values and zero/nonzero status
    const outRows = state.rowsA;
    const outCols = state.matrixCount === 2 ? state.colsB : state.colsC;
    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        const valElem = document.getElementById(`val-C-${r}-${c}`);
        const cellC = document.getElementById(`cell-C-${r}-${c}`);
        const val = state.matrixC[r]?.[c] ?? 0;
        if (valElem) {
          valElem.textContent = val;
        }
        if (cellC) {
          updateCellZeroState(cellC, val);
        }
      }
    }
    // Update step view
    if (state.currentStepIndex >= 0) {
      setStepIndex(Math.min(state.currentStepIndex, state.steps.length - 1));
    } else {
      updateInspectorUI();
    }
  }

  // --- Stepper & Playback Engine ---
  function clearAllHighlights() {
    document.querySelectorAll('.matrix-cell').forEach(c => {
      c.classList.remove('row-active', 'col-active', 'term-active', 'target-active');
    });
  }

  function setStepIndex(index) {
    if (!canMultiply() || state.steps.length === 0) {
      clearAllHighlights();
      updateProgressUI(0, 0);
      return;
    }

    state.currentStepIndex = Math.max(-1, Math.min(index, state.steps.length - 1));
    clearAllHighlights();

    if (state.currentStepIndex === -1) {
      // Idle beginning state
      updateProgressUI(0, state.steps.length);
      el.currentCellName.textContent = 'Ready';
      el.ruleRow.textContent = '-';
      el.ruleCol.textContent = '-';
      el.termsBreakdownRow.innerHTML = '<span style="color:var(--text-dim);font-size:0.85rem;">Click Play, Step, or any cell in Result to start inspecting.</span>';
      el.formulaBody.textContent = 'None';
      el.runningTotalValue.textContent = '0';

      // Restore calculated values in Intermediate grid if 3-matrix mode
      if (state.matrixCount === 3) {
        for (let r = 0; r < state.rowsA; r++) {
          for (let c = 0; c < state.colsB; c++) {
            const valElem = document.getElementById(`val-Inter-${r}-${c}`);
            const cellInter = document.getElementById(`cell-Inter-${r}-${c}`);
            const val = state.intermediateAB[r]?.[c] ?? 0;
            if (valElem) valElem.textContent = val;
            if (cellInter) updateCellZeroState(cellInter, val);
          }
        }
      }

      // Restore calculated values in Result grid
      const outRows = state.rowsA;
      const outCols = state.matrixCount === 2 ? state.colsB : state.colsC;
      for (let r = 0; r < outRows; r++) {
        for (let c = 0; c < outCols; c++) {
          const valElem = document.getElementById(`val-C-${r}-${c}`);
          const cellC = document.getElementById(`cell-C-${r}-${c}`);
          const val = state.matrixC[r]?.[c] ?? 0;
          if (valElem) valElem.textContent = val;
          if (cellC) updateCellZeroState(cellC, val);
        }
      }
      return;
    }

    const currentStep = state.steps[state.currentStepIndex];
    const { stage, i, j, k, valA, valB, prod, runningSum, isCellEnd, cellTerms, cellTotalTerms, targetLabel } = currentStep;

    if (state.matrixCount === 2 || stage === 1) {
      // Stage 1 (A x B)
      // 1. Highlight Row i in Matrix A
      for (let colIdx = 0; colIdx < state.colsA; colIdx++) {
        const cellA = document.getElementById(`cell-A-${i}-${colIdx}`);
        if (cellA) cellA.classList.add('row-active');
      }

      // 2. Highlight Column j in Matrix B
      for (let rowIdx = 0; rowIdx < state.rowsB; rowIdx++) {
        const cellB = document.getElementById(`cell-B-${rowIdx}-${j}`);
        if (cellB) cellB.classList.add('col-active');
      }

      // 3. Highlight exact multiplying pair A[i][k] and B[k][j]
      const termCellA = document.getElementById(`cell-A-${i}-${k}`);
      const termCellB = document.getElementById(`cell-B-${k}-${j}`);
      if (termCellA) termCellA.classList.add('term-active');
      if (termCellB) termCellB.classList.add('term-active');

      // 4. Highlight target cell
      if (state.matrixCount === 2) {
        const targetCellC = document.getElementById(`cell-C-${i}-${j}`);
        if (targetCellC) {
          targetCellC.classList.add('target-active', 'target-computed');
          const valElem = document.getElementById(`val-C-${i}-${j}`);
          if (valElem) valElem.textContent = runningSum;
          updateCellZeroState(targetCellC, runningSum);
        }
        el.currentCellName.textContent = `C[${i + 1}, ${j + 1}]`;
        el.ruleRow.textContent = `${i + 1}`;
        el.ruleCol.textContent = `${j + 1}`;
      } else {
        // In 3-matrix mode Stage 1 targets Intermediate (A x B)
        const targetInter = document.getElementById(`cell-Inter-${i}-${j}`);
        if (targetInter) {
          targetInter.classList.add('target-active', 'target-computed');
          const valElem = document.getElementById(`val-Inter-${i}-${j}`);
          if (valElem) valElem.textContent = runningSum;
          updateCellZeroState(targetInter, runningSum);
        }
        el.currentCellName.textContent = `(A×B)[${i + 1}, ${j + 1}] (Stage 1)`;
        el.ruleRow.textContent = `Row ${i + 1} (A)`;
        el.ruleCol.textContent = `Col ${j + 1} (B)`;
      }
    } else {
      // Stage 2 (Intermediate AB x C -> Result R)
      // 1. Highlight Row i in Intermediate Product
      for (let colIdx = 0; colIdx < state.colsB; colIdx++) {
        const cellInter = document.getElementById(`cell-Inter-${i}-${colIdx}`);
        if (cellInter) cellInter.classList.add('row-active');
      }

      // 2. Highlight Column j in Matrix C Input
      for (let rowIdx = 0; rowIdx < state.rowsC; rowIdx++) {
        const cellCIn = document.getElementById(`cell-CIn-${rowIdx}-${j}`);
        if (cellCIn) cellCIn.classList.add('col-active');
      }

      // 3. Highlight exact multiplying pair AB[i][k] and C[k][j]
      const termCellInter = document.getElementById(`cell-Inter-${i}-${k}`);
      const termCellCIn = document.getElementById(`cell-CIn-${k}-${j}`);
      if (termCellInter) termCellInter.classList.add('term-active');
      if (termCellCIn) termCellCIn.classList.add('term-active');

      // 4. Highlight target cell in Final Result
      const targetCellC = document.getElementById(`cell-C-${i}-${j}`);
      if (targetCellC) {
        targetCellC.classList.add('target-active', 'target-computed');
        const valElem = document.getElementById(`val-C-${i}-${j}`);
        if (valElem) valElem.textContent = runningSum;
        updateCellZeroState(targetCellC, runningSum);
      }
      el.currentCellName.textContent = `Result R[${i + 1}, ${j + 1}] (Stage 2)`;
      el.ruleRow.textContent = `Row ${i + 1} (A×B)`;
      el.ruleCol.textContent = `Col ${j + 1} (C)`;
    }

    // Render terms breakdown chips
    el.termsBreakdownRow.innerHTML = '';
    for (let t = 0; t < cellTotalTerms; t++) {
      const termChip = document.createElement('div');
      termChip.className = 'term-chip';
      
      let aVal = 0;
      let bVal = 0;
      if (state.matrixCount === 2 || stage === 1) {
        aVal = state.matrixA[i]?.[t] ?? 0;
        bVal = state.matrixB[t]?.[j] ?? 0;
      } else {
        aVal = state.intermediateAB[i]?.[t] ?? 0;
        bVal = state.matrixCInput[t]?.[j] ?? 0;
      }
      const pVal = aVal * bVal;

      if (t < k) {
        termChip.classList.add('done');
      } else if (t === k) {
        termChip.classList.add('active');
      }

      termChip.innerHTML = `
        <span class="val-a">${aVal}</span>
        <span class="op-times">&times;</span>
        <span class="val-b">${bVal}</span>
        <span class="op-times">=</span>
        <span class="val-prod">${pVal}</span>
      `;

      // Allow clicking a term chip to jump directly to this term!
      termChip.addEventListener('click', () => {
        const stepTarget = state.steps.findIndex(s => s.stage === stage && s.i === i && s.j === j && s.k === t);
        if (stepTarget !== -1) {
          setStepIndex(stepTarget);
        }
      });

      el.termsBreakdownRow.appendChild(termChip);
    }

    // Formula accumulation string
    const completedParts = cellTerms.map(t => `(${t.valA} &times; ${t.valB})`);
    const remainingPartsCount = cellTotalTerms - cellTerms.length;
    let formulaText = completedParts.join(' + ');
    if (remainingPartsCount > 0) {
      formulaText += ` + ... (${remainingPartsCount} term${remainingPartsCount > 1 ? 's' : ''} left)`;
    }
    el.formulaBody.innerHTML = formulaText;
    el.runningTotalValue.textContent = runningSum;

    // Narrative text
    const stagePrefix = state.matrixCount === 3 ? `[Stage ${stage} of 2] ` : '';
    if (isCellEnd) {
      el.narrativeExplanation.innerHTML = `${stagePrefix}Completed cell <strong>${targetLabel}</strong>! Sum is <strong>${runningSum}</strong>.`;
    } else {
      el.narrativeExplanation.innerHTML = `${stagePrefix}Multiplying (k=${k + 1}: <strong>${valA}</strong>) &times; (k=${k + 1}: <strong>${valB}</strong>) = <strong>${prod}</strong>. Running accumulator: <strong>${runningSum}</strong>.`;
    }

    // Progress UI
    updateProgressUI(state.currentStepIndex + 1, state.steps.length, `${targetLabel} (k=${k + 1}/${cellTotalTerms})`);

    // Audio sound
    sfx.playStepSound(k, cellTotalTerms);

    // If reached end
    if (state.currentStepIndex === state.steps.length - 1) {
      sfx.playCompletionChime();
      if (state.isPlaying) {
        stopPlaying();
      }
    }
  }

  function updateProgressUI(current, total, targetName = '') {
    el.stepCounterText.textContent = `Step ${current} of ${total}`;
    el.stepTargetText.textContent = targetName ? `Active: ${targetName}` : 'Idle';
    const percent = total > 0 ? Math.round((current / total) * 100) : 0;
    el.stepPercentText.textContent = `${percent}%`;
    el.progressBarFill.style.width = `${percent}%`;

    el.btnPrev.disabled = current <= 0;
    el.btnFirst.disabled = current <= 0;
    el.btnNext.disabled = current >= total;
    el.btnLast.disabled = current >= total;
  }

  function updateInspectorUI() {
    if (state.currentStepIndex >= 0 && state.currentStepIndex < state.steps.length) {
      setStepIndex(state.currentStepIndex);
    } else {
      setStepIndex(-1);
    }
  }

  // Next / Prev step navigation respecting 'mode' (subterm or cell)
  function stepForward() {
    if (!canMultiply() || state.steps.length === 0) return;

    if (state.stepMode === 'subterm') {
      if (state.currentStepIndex < state.steps.length - 1) {
        setStepIndex(state.currentStepIndex + 1);
      } else {
        stopPlaying();
      }
    } else {
      // Cell mode: jump to next cell's last subterm (full cell completed)
      const cur = state.steps[state.currentStepIndex];
      let targetIndex = -1;

      if (!cur) {
        // Find end of first cell
        targetIndex = state.steps.findIndex(s => s.isCellEnd);
      } else {
        // Find next cell end
        targetIndex = state.steps.findIndex((s, idx) => idx > state.currentStepIndex && s.isCellEnd);
      }

      if (targetIndex !== -1) {
        setStepIndex(targetIndex);
      } else if (state.currentStepIndex < state.steps.length - 1) {
        setStepIndex(state.steps.length - 1);
      } else {
        stopPlaying();
      }
    }
  }

  function stepBackward() {
    if (!canMultiply() || state.steps.length === 0) return;

    if (state.stepMode === 'subterm') {
      if (state.currentStepIndex > 0) {
        setStepIndex(state.currentStepIndex - 1);
      } else {
        setStepIndex(-1);
      }
    } else {
      // Cell mode: jump backward to previous cell's end
      const cur = state.steps[state.currentStepIndex];
      if (!cur || state.currentStepIndex <= 0) {
        setStepIndex(-1);
        return;
      }
      
      const curCellI = cur.i;
      const curCellJ = cur.j;

      // Find the end of previous cell
      for (let idx = state.currentStepIndex - 1; idx >= 0; idx--) {
        const s = state.steps[idx];
        if (s.isCellEnd && (s.i !== curCellI || s.j !== curCellJ)) {
          setStepIndex(idx);
          return;
        }
      }
      setStepIndex(-1);
    }
  }

  function jumpToFirst() {
    setStepIndex(-1);
  }

  function jumpToLast() {
    if (!canMultiply() || state.steps.length === 0) return;
    setStepIndex(state.steps.length - 1);
  }

  function jumpToCell(r, c) {
    if (!canMultiply() || state.steps.length === 0) return;
    stopPlaying();
    // Find the end step of cell C[r, c]
    const stepTarget = state.steps.findIndex(s => s.i === r && s.j === c && s.isCellEnd);
    if (stepTarget !== -1) {
      setStepIndex(stepTarget);
    }
  }

  // --- Auto-play Loop ---
  function togglePlay() {
    if (state.isPlaying) {
      stopPlaying();
    } else {
      startPlaying();
    }
  }

  function startPlaying() {
    if (!canMultiply()) return;

    // Auto-fill blanks with 0 if any exist
    const hasEmpty = state.matrixA.some(row => row.some(v => v === '' || v === null || (typeof v === 'number' && isNaN(v)))) ||
                     state.matrixB.some(row => row.some(v => v === '' || v === null || (typeof v === 'number' && isNaN(v)))) ||
                     (state.matrixCount === 3 && state.matrixCInput.some(row => row.some(v => v === '' || v === null || (typeof v === 'number' && isNaN(v)))));
    if (hasEmpty) {
      calculateAndFillZeros();
    }

    if (state.steps.length === 0) return;
    
    // If at end, loop back to start
    if (state.currentStepIndex >= state.steps.length - 1) {
      setStepIndex(-1);
    }

    state.isPlaying = true;
    el.playPauseText.textContent = 'Pause';
    el.iconPlay.classList.add('hidden');
    el.iconPause.classList.remove('hidden');

    runPlayLoop();
  }

  function stopPlaying() {
    state.isPlaying = false;
    if (state.playTimer) {
      clearTimeout(state.playTimer);
      state.playTimer = null;
    }
    el.playPauseText.textContent = 'Play';
    el.iconPlay.classList.remove('hidden');
    el.iconPause.classList.add('hidden');
  }

  function runPlayLoop() {
    if (!state.isPlaying) return;

    stepForward();

    if (state.isPlaying && state.currentStepIndex < state.steps.length - 1) {
      const baseDelay = state.stepMode === 'cell' ? 1200 : 700;
      const delay = Math.max(120, baseDelay / state.playSpeed);
      state.playTimer = setTimeout(runPlayLoop, delay);
    } else {
      stopPlaying();
    }
  }

  // --- Dimension Change Handlers ---
  function updateDimensions(newRowsA, newColsA, newRowsB, newColsB) {
    stopPlaying();

    const clamp = (val) => Math.max(1, Math.min(10, parseInt(val, 10) || 1));

    state.rowsA = clamp(newRowsA);
    state.colsA = clamp(newColsA);
    state.rowsB = clamp(newRowsB);
    state.colsB = clamp(newColsB);

    if (state.isLinked) {
      state.rowsB = state.colsA;
      if (state.matrixCount === 3) {
        state.rowsC = state.colsB;
        state.matrixCInput = resizeMatrixPreserving(state.matrixCInput, state.rowsC, state.colsC);
      }
    }

    // Resize matrices while preserving existing numbers
    state.matrixA = resizeMatrixPreserving(state.matrixA, state.rowsA, state.colsA);
    state.matrixB = resizeMatrixPreserving(state.matrixB, state.rowsB, state.colsB);

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);
  }

  function updateDimensionsC(newRowsC, newColsC) {
    stopPlaying();
    const clamp = (val) => Math.max(1, Math.min(10, parseInt(val, 10) || 1));

    state.rowsC = clamp(newRowsC);
    state.colsC = clamp(newColsC);

    if (state.isLinked) {
      state.colsB = state.rowsC;
      state.matrixB = resizeMatrixPreserving(state.matrixB, state.rowsB, state.colsB);
    }

    state.matrixCInput = resizeMatrixPreserving(state.matrixCInput, state.rowsC, state.colsC);

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);
  }

  function resizeMatrixPreserving(oldMat, newRows, newCols) {
    const newMat = [];
    for (let r = 0; r < newRows; r++) {
      const row = [];
      for (let c = 0; c < newCols; c++) {
        const existingVal = oldMat?.[r]?.[c];
        row.push(existingVal !== undefined ? existingVal : randomInt(-5, 9));
      }
      newMat.push(row);
    }
    return newMat;
  }

  // --- 2-Matrix vs 3-Matrix Mode Switcher ---
  function setMatrixCount(count) {
    stopPlaying();
    state.matrixCount = count;

    if (count === 3) {
      el.mode2MatricesBtn.classList.remove('active');
      el.mode3MatricesBtn.classList.add('active');
      el.dimBoxMatrixC.classList.remove('hidden');
      el.opTimesBC.classList.remove('hidden');
      el.cardMatrixCInput.classList.remove('hidden');
      el.cardIntermediate.classList.remove('hidden');
      if (el.tabMatrixCInput) el.tabMatrixCInput.classList.remove('hidden');

      if (el.labelColsB) el.labelColsB.textContent = 'Cols (l)';
      if (el.resCardTitle) el.resCardTitle.textContent = 'Final Result R';
      if (el.resCardHint) el.resCardHint.textContent = 'R = (A × B) × C';
      if (el.resLabelBadge) el.resLabelBadge.textContent = 'Matrix R';

      // Ensure inner dimension matching between B and C
      state.rowsC = state.colsB;
      if (!state.colsC) state.colsC = 2;
      if (!state.matrixCInput || state.matrixCInput.length !== state.rowsC || (state.matrixCInput[0] && state.matrixCInput[0].length !== state.colsC)) {
        state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => randomInt(-5, 9));
      }
    } else {
      el.mode2MatricesBtn.classList.add('active');
      el.mode3MatricesBtn.classList.remove('active');
      el.dimBoxMatrixC.classList.add('hidden');
      el.opTimesBC.classList.add('hidden');
      el.cardMatrixCInput.classList.add('hidden');
      el.cardIntermediate.classList.add('hidden');
      if (el.tabMatrixCInput) el.tabMatrixCInput.classList.add('hidden');

      if (el.labelColsB) el.labelColsB.textContent = 'Cols (n)';
      if (el.resCardTitle) el.resCardTitle.textContent = 'Result Matrix C';
      if (el.resCardHint) el.resCardHint.textContent = 'C = A × B';
      if (el.resLabelBadge) el.resLabelBadge.textContent = 'Matrix C';
    }

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);

    el.narrativeExplanation.innerHTML = count === 3
      ? `Switched to <strong>3-Matrix Mode</strong>: <code>(A &times; B) &times; C = R</code>. Step through both stages (intermediate product & final result)!`
      : `Switched to <strong>2-Matrix Mode</strong>: <code>A &times; B = C</code>.`;
  }

  // --- Swap Matrix A and Matrix B ---
  function swapMatrixAandB() {
    stopPlaying();

    const tempRows = state.rowsA;
    const tempCols = state.colsA;
    const tempMat = state.matrixA;

    state.rowsA = state.rowsB;
    state.colsA = state.colsB;
    state.matrixA = state.matrixB;

    state.rowsB = tempRows;
    state.colsB = tempCols;
    state.matrixB = tempMat;

    if (state.isLinked) {
      state.rowsB = state.colsA;
      state.matrixB = resizeMatrixPreserving(state.matrixB, state.rowsB, state.colsB);
      if (state.matrixCount === 3) {
        state.rowsC = state.colsB;
        state.matrixCInput = resizeMatrixPreserving(state.matrixCInput, state.rowsC, state.colsC);
      }
    }

    // Trigger visual swap animation on cards
    if (el.cardMatrixA) {
      el.cardMatrixA.classList.remove('animate-swap');
      void el.cardMatrixA.offsetWidth;
      el.cardMatrixA.classList.add('animate-swap');
    }
    if (el.cardMatrixB) {
      el.cardMatrixB.classList.remove('animate-swap');
      void el.cardMatrixB.offsetWidth;
      el.cardMatrixB.classList.add('animate-swap');
    }
    setTimeout(() => {
      el.cardMatrixA?.classList.remove('animate-swap');
      el.cardMatrixB?.classList.remove('animate-swap');
    }, 500);

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);

    sfx.playStepSound(0, 1);
    el.narrativeExplanation.innerHTML = `Swapped Matrix A and Matrix B (<strong>A ⇄ B</strong>). Dimensions and cell values exchanged.`;
  }

  // --- Dimension Presets ---
  const PRESETS = {
    '2x2': { rowsA: 2, colsA: 2, rowsB: 2, colsB: 2 },
    '3x3': { rowsA: 3, colsA: 3, rowsB: 3, colsB: 3 },
    '2x3_3x2': { rowsA: 2, colsA: 3, rowsB: 3, colsB: 2 },
    '3x4_4x2': { rowsA: 3, colsA: 4, rowsB: 4, colsB: 2 },
    '4x4': { rowsA: 4, colsA: 4, rowsB: 4, colsB: 4 },
    '1x4_4x1': { rowsA: 1, colsA: 4, rowsB: 4, colsB: 1 }
  };

  function applyPreset(rowsA, colsA, rowsB, colsB) {
    state.rowsA = rowsA;
    state.colsA = colsA;
    state.rowsB = rowsB;
    state.colsB = colsB;
    state.matrixA = createMatrix(rowsA, colsA, () => randomInt(-5, 9));
    state.matrixB = createMatrix(rowsB, colsB, () => randomInt(-5, 9));
    if (state.matrixCount === 3) {
      state.rowsC = colsB;
      state.colsC = 2;
      state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => randomInt(-5, 9));
    }
    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);
  }

  // --- Fill Handlers ---
  function fillRandomAll() {
    state.matrixA = createMatrix(state.rowsA, state.colsA, () => randomInt(-9, 9));
    state.matrixB = createMatrix(state.rowsB, state.colsB, () => randomInt(-9, 9));
    if (state.matrixCount === 3) {
      state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => randomInt(-9, 9));
      renderGridCInput();
    }
    onMatrixDataChanged();
    renderGridA();
    renderGridB();
  }

  function clearAllZeros() {
    state.matrixA = createMatrix(state.rowsA, state.colsA, () => 0);
    state.matrixB = createMatrix(state.rowsB, state.colsB, () => 0);
    if (state.matrixCount === 3) {
      state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => 0);
      renderGridCInput();
    }
    onMatrixDataChanged();
    renderGridA();
    renderGridB();
  }

  // --- Empty Matrices & Calculate ---
  function emptyMatrixA() {
    state.matrixA = createMatrix(state.rowsA, state.colsA, () => '');
    renderGridA();
    onMatrixDataChanged();
  }

  function emptyMatrixB() {
    state.matrixB = createMatrix(state.rowsB, state.colsB, () => '');
    renderGridB();
    onMatrixDataChanged();
  }

  function emptyMatrixC() {
    state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => '');
    renderGridCInput();
    onMatrixDataChanged();
  }

  function randMatrixC() {
    state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => randomInt(-9, 9));
    renderGridCInput();
    onMatrixDataChanged();
  }

  function clearMatrixC() {
    state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => 0);
    renderGridCInput();
    onMatrixDataChanged();
  }

  function emptyAllMatrices() {
    stopPlaying();
    state.matrixA = createMatrix(state.rowsA, state.colsA, () => '');
    state.matrixB = createMatrix(state.rowsB, state.colsB, () => '');
    if (state.matrixCount === 3) {
      state.matrixCInput = createMatrix(state.rowsC, state.colsC, () => '');
      renderGridCInput();
    }
    renderGridA();
    renderGridB();
    onMatrixDataChanged();
    setStepIndex(-1);
    el.narrativeExplanation.innerHTML = `All matrices cleared to <strong>empty boxes</strong>. Enter your custom numbers or click <strong>Calculate (Fill 0s)</strong> to auto-fill blanks with 0.`;
  }

  function calculateAndFillZeros() {
    stopPlaying();
    let filledCount = 0;

    // 1. Fill empty boxes in Matrix A with 0
    for (let r = 0; r < state.rowsA; r++) {
      for (let c = 0; c < state.colsA; c++) {
        const val = state.matrixA[r]?.[c];
        if (val === '' || val === null || val === undefined || (typeof val === 'number' && isNaN(val))) {
          state.matrixA[r][c] = 0;
          filledCount++;
          const cellA = document.getElementById(`cell-A-${r}-${c}`);
          if (cellA) {
            const input = cellA.querySelector('input');
            if (input) input.value = 0;
            updateCellZeroState(cellA, 0);
            cellA.classList.add('fill-flash');
            setTimeout(() => cellA.classList.remove('fill-flash'), 450);
          }
        }
      }
    }

    // 2. Fill empty boxes in Matrix B with 0
    for (let r = 0; r < state.rowsB; r++) {
      for (let c = 0; c < state.colsB; c++) {
        const val = state.matrixB[r]?.[c];
        if (val === '' || val === null || val === undefined || (typeof val === 'number' && isNaN(val))) {
          state.matrixB[r][c] = 0;
          filledCount++;
          const cellB = document.getElementById(`cell-B-${r}-${c}`);
          if (cellB) {
            const input = cellB.querySelector('input');
            if (input) input.value = 0;
            updateCellZeroState(cellB, 0);
            cellB.classList.add('fill-flash');
            setTimeout(() => cellB.classList.remove('fill-flash'), 450);
          }
        }
      }
    }

    // 3. Fill empty boxes in Matrix C (Input) with 0 if 3-matrix mode
    if (state.matrixCount === 3) {
      for (let r = 0; r < state.rowsC; r++) {
        for (let c = 0; c < state.colsC; c++) {
          const val = state.matrixCInput[r]?.[c];
          if (val === '' || val === null || val === undefined || (typeof val === 'number' && isNaN(val))) {
            state.matrixCInput[r][c] = 0;
            filledCount++;
            const cellCIn = document.getElementById(`cell-CIn-${r}-${c}`);
            if (cellCIn) {
              const input = cellCIn.querySelector('input');
              if (input) input.value = 0;
              updateCellZeroState(cellCIn, 0);
              cellCIn.classList.add('fill-flash');
              setTimeout(() => cellCIn.classList.remove('fill-flash'), 450);
            }
          }
        }
      }
    }

    // 4. Compute Result Matrix and generate steps
    computeResultMatrix();
    generateSteps();

    // 5. Update Intermediate display if 3-matrix mode
    if (state.matrixCount === 3) {
      for (let r = 0; r < state.rowsA; r++) {
        for (let c = 0; c < state.colsB; c++) {
          const valElem = document.getElementById(`val-Inter-${r}-${c}`);
          const cellInter = document.getElementById(`cell-Inter-${r}-${c}`);
          const val = state.intermediateAB[r]?.[c] ?? 0;
          if (valElem) valElem.textContent = val;
          if (cellInter) updateCellZeroState(cellInter, val);
        }
      }
    }

    // 6. Update Final Result display
    const outRows = state.rowsA;
    const outCols = state.matrixCount === 2 ? state.colsB : state.colsC;
    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        const valElem = document.getElementById(`val-C-${r}-${c}`);
        const cellC = document.getElementById(`cell-C-${r}-${c}`);
        const val = state.matrixC[r]?.[c] ?? 0;
        if (valElem) valElem.textContent = val;
        if (cellC) updateCellZeroState(cellC, val);
      }
    }

    setStepIndex(-1);

    // Audio cue
    sfx.playStepSound(0, 1);

    // Narrative feedback
    const resName = state.matrixCount === 2 ? 'Result Matrix C' : 'Final Result R';
    if (filledCount > 0) {
      el.narrativeExplanation.innerHTML = `Filled <strong>${filledCount} empty box${filledCount > 1 ? 'es' : ''}</strong> with <strong>0</strong> and calculated ${resName}.`;
    } else {
      el.narrativeExplanation.innerHTML = `Calculated ${resName} successfully. Ready for step-by-step playback or inspection.`;
    }
  }

  function fillIdentityAll() {
    state.matrixA = createMatrix(state.rowsA, state.colsA, (r, c) => (r === c ? 1 : 0));
    state.matrixB = createMatrix(state.rowsB, state.colsB, (r, c) => (r === c ? 1 : 0));
    if (state.matrixCount === 3) {
      state.matrixCInput = createMatrix(state.rowsC, state.colsC, (r, c) => (r === c ? 1 : 0));
      renderGridCInput();
    }
    onMatrixDataChanged();
    renderGridA();
    renderGridB();
  }

  // --- Export Formatting ---
  function generateExportString(format) {
    if (!canMultiply() || state.matrixC.length === 0) {
      return 'No valid result matrix computed.';
    }

    const C = state.matrixC;
    const m = state.rowsA;
    const n = state.colsB;

    switch (format) {
      case 'latex': {
        const rowsStr = C.map(row => row.join(' & ')).join(' \\\\\n  ');
        return `\\begin{pmatrix}\n  ${rowsStr}\n\\end{pmatrix}`;
      }
      case 'json': {
        return JSON.stringify(C, null, 2);
      }
      case 'csv': {
        return C.map(row => row.join(',')).join('\n');
      }
      case 'python': {
        const rowsStr = C.map(row => `    [${row.join(', ')}]`).join(',\n');
        return `import numpy as np\n\nC = np.array([\n${rowsStr}\n])`;
      }
      default:
        return '';
    }
  }

  function openExportModal() {
    el.exportContentArea.value = generateExportString(activeExportFormat);
    el.copyStatusText.textContent = '';
    el.exportModal.classList.remove('hidden');
    state.activeModal = el.exportModal;
  }

  function closeExportModal() {
    el.exportModal.classList.add('hidden');
    state.activeModal = null;
  }

  // --- Batch / Paste Modal ---
  function openPasteModal() {
    let targetMatrix = state.matrixA;
    if (activePasteTarget === 'B') targetMatrix = state.matrixB;
    else if (activePasteTarget === 'C') targetMatrix = state.matrixCInput;

    el.matrixRawInput.value = targetMatrix.map(row => row.join('\t')).join('\n');
    el.pasteModal.classList.remove('hidden');
    state.activeModal = el.pasteModal;
    el.matrixRawInput.focus();
  }

  function closePasteModal() {
    el.pasteModal.classList.add('hidden');
    state.activeModal = null;
  }

  function applyParsedPaste() {
    const raw = el.matrixRawInput.value.trim();
    if (!raw) return;

    const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const parsed = lines.map(line => {
      return line.split(/[\s,]+/).map(v => parseFloat(v)).filter(v => Number.isFinite(v));
    }).filter(row => row.length > 0);

    if (parsed.length === 0) return;

    const MAX_DIM = 10;
    const rawRows = parsed.length;
    const rawCols = Math.max(...parsed.map(r => r.length));
    const newRows = Math.min(MAX_DIM, rawRows);
    const newCols = Math.min(MAX_DIM, rawCols);

    // Pad ragged rows with 0 and clamp to MAX_DIM
    const normalized = parsed.slice(0, newRows).map(row => {
      const clampedRow = row.slice(0, newCols);
      while (clampedRow.length < newCols) clampedRow.push(0);
      return clampedRow;
    });

    if (activePasteTarget === 'A') {
      state.rowsA = newRows;
      state.colsA = newCols;
      if (state.isLinked) {
        state.rowsB = newCols;
        state.matrixB = resizeMatrixPreserving(state.matrixB, state.rowsB, state.colsB);
        if (state.matrixCount === 3) {
          state.rowsC = state.colsB;
          state.matrixCInput = resizeMatrixPreserving(state.matrixCInput, state.rowsC, state.colsC);
        }
      }
      state.matrixA = normalized;
    } else if (activePasteTarget === 'B') {
      state.rowsB = newRows;
      state.colsB = newCols;
      if (state.isLinked) {
        state.colsA = newRows;
        state.matrixA = resizeMatrixPreserving(state.matrixA, state.rowsA, state.colsA);
        if (state.matrixCount === 3) {
          state.rowsC = state.colsB;
          state.matrixCInput = resizeMatrixPreserving(state.matrixCInput, state.rowsC, state.colsC);
        }
      }
      state.matrixB = normalized;
    } else if (activePasteTarget === 'C') {
      state.rowsC = newRows;
      state.colsC = newCols;
      if (state.isLinked) {
        state.colsB = newRows;
        state.matrixB = resizeMatrixPreserving(state.matrixB, state.rowsB, state.colsB);
      }
      state.matrixCInput = normalized;
    }

    computeResultMatrix();
    generateSteps();
    renderAll();
    setStepIndex(-1);
    closePasteModal();

    if (rawRows > MAX_DIM || rawCols > MAX_DIM) {
      el.narrativeExplanation.innerHTML = `Matrix parsed and clamped to <strong>${newRows} &times; ${newCols}</strong> (maximum supported dimension is 10 &times; 10 for performance safety).`;
    }
  }

  // --- Theme Switcher (Dark / Light) ---
  function getPreferredTheme() {
    try {
      const saved = localStorage.getItem('matrixflow-theme');
      if (saved === 'dark' || saved === 'light') return saved;
    } catch (e) {}
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const isLight = theme === 'light';
    if (el.themeSunIcon && el.themeMoonIcon) {
      el.themeSunIcon.classList.toggle('hidden', isLight);
      el.themeMoonIcon.classList.toggle('hidden', !isLight);
    }
    try {
      localStorage.setItem('matrixflow-theme', theme);
    } catch (e) {}
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
    const next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
  }

  // --- Event Listeners Registration ---
  function setupEventListeners() {
    // Dimension inputs
    el.rowsA.addEventListener('change', () => updateDimensions(el.rowsA.value, state.colsA, state.rowsB, state.colsB));
    el.colsA.addEventListener('change', () => updateDimensions(state.rowsA, el.colsA.value, state.isLinked ? el.colsA.value : state.rowsB, state.colsB));
    el.rowsB.addEventListener('change', () => updateDimensions(state.rowsA, state.isLinked ? el.rowsB.value : state.colsA, el.rowsB.value, state.colsB));
    el.colsB.addEventListener('change', () => updateDimensions(state.rowsA, state.colsA, state.rowsB, el.colsB.value));

    // Stepper buttons
    el.decRowsA.addEventListener('click', () => updateDimensions(state.rowsA - 1, state.colsA, state.rowsB, state.colsB));
    el.incRowsA.addEventListener('click', () => updateDimensions(state.rowsA + 1, state.colsA, state.rowsB, state.colsB));
    el.decColsA.addEventListener('click', () => updateDimensions(state.rowsA, state.colsA - 1, state.isLinked ? state.colsA - 1 : state.rowsB, state.colsB));
    el.incColsA.addEventListener('click', () => updateDimensions(state.rowsA, state.colsA + 1, state.isLinked ? state.colsA + 1 : state.rowsB, state.colsB));
    el.decRowsB.addEventListener('click', () => updateDimensions(state.rowsA, state.isLinked ? state.rowsB - 1 : state.colsA, state.rowsB - 1, state.colsB));
    el.incRowsB.addEventListener('click', () => updateDimensions(state.rowsA, state.isLinked ? state.rowsB + 1 : state.colsA, state.rowsB + 1, state.colsB));
    el.decColsB.addEventListener('click', () => updateDimensions(state.rowsA, state.colsA, state.rowsB, state.colsB - 1));
    el.incColsB.addEventListener('click', () => updateDimensions(state.rowsA, state.colsA, state.rowsB, state.colsB + 1));

    // Sync button toggle
    el.syncDimBtn.addEventListener('click', () => {
      state.isLinked = !state.isLinked;
      el.syncDimBtn.classList.toggle('active', state.isLinked);
      if (state.isLinked && state.colsA !== state.rowsB) {
        updateDimensions(state.rowsA, state.colsA, state.colsA, state.colsB);
      }
    });

    el.fixDimensionBtn.addEventListener('click', () => {
      updateDimensions(state.rowsA, state.colsA, state.colsA, state.colsB);
    });

    // Player controls
    el.btnPlayPause.addEventListener('click', togglePlay);
    el.btnNext.addEventListener('click', () => { stopPlaying(); stepForward(); });
    el.btnPrev.addEventListener('click', () => { stopPlaying(); stepBackward(); });
    el.btnFirst.addEventListener('click', () => { stopPlaying(); jumpToFirst(); });
    el.btnLast.addEventListener('click', () => { stopPlaying(); jumpToLast(); });
    el.btnReset.addEventListener('click', () => { stopPlaying(); setStepIndex(-1); });

    // Step mode segmented buttons
    el.modeSubTerm.addEventListener('click', () => {
      state.stepMode = 'subterm';
      el.modeSubTerm.classList.add('active');
      el.modeCell.classList.remove('active');
    });

    el.modeCell.addEventListener('click', () => {
      state.stepMode = 'cell';
      el.modeCell.classList.add('active');
      el.modeSubTerm.classList.remove('active');
    });

    // Speed range
    el.speedRange.addEventListener('input', (e) => {
      state.playSpeed = parseFloat(e.target.value);
      el.speedValueText.textContent = `${state.playSpeed.toFixed(1)}×`;
    });

    // Sound toggle
    el.soundToggleBtn.addEventListener('click', () => {
      sfx.enabled = !sfx.enabled;
      el.soundOnIcon.classList.toggle('hidden', !sfx.enabled);
      el.soundOffIcon.classList.toggle('hidden', sfx.enabled);
    });

    // Theme toggle (Dark / Light)
    el.themeToggleBtn?.addEventListener('click', toggleTheme);

    // Watch system color scheme changes when no manual override is saved
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
        try {
          if (!localStorage.getItem('matrixflow-theme')) {
            applyTheme(e.matches ? 'light' : 'dark');
          }
        } catch (err) {}
      });
    }

    // Presets dropdown
    el.presetBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      el.presetMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', (e) => {
      if (!el.presetBtn.contains(e.target) && !el.presetMenu.contains(e.target)) {
        el.presetMenu.classList.add('hidden');
      }
    });

    el.presetMenu.querySelectorAll('.dropdown-item').forEach(btn => {
      btn.addEventListener('click', () => {
        const rA = parseInt(btn.dataset.rowsA, 10);
        const cA = parseInt(btn.dataset.colsA, 10);
        const rB = parseInt(btn.dataset.rowsB, 10);
        const cB = parseInt(btn.dataset.colsB, 10);
        applyPreset(rA, cA, rB, cB);
        el.presetMenu.classList.add('hidden');
      });
    });

    // Fill and Calculate buttons
    el.randomizeAllBtn.addEventListener('click', fillRandomAll);
    el.clearAllBtn.addEventListener('click', clearAllZeros);
    el.emptyAllBtn.addEventListener('click', emptyAllMatrices);
    el.calculateBtn.addEventListener('click', calculateAndFillZeros);
    el.identityBtn.addEventListener('click', fillIdentityAll);

    el.randABtn.addEventListener('click', () => {
      state.matrixA = createMatrix(state.rowsA, state.colsA, () => randomInt(-9, 9));
      onMatrixDataChanged();
      renderGridA();
    });

    el.clearABtn.addEventListener('click', () => {
      state.matrixA = createMatrix(state.rowsA, state.colsA, () => 0);
      onMatrixDataChanged();
      renderGridA();
    });

    el.emptyABtn.addEventListener('click', emptyMatrixA);

    el.randBBtn.addEventListener('click', () => {
      state.matrixB = createMatrix(state.rowsB, state.colsB, () => randomInt(-9, 9));
      onMatrixDataChanged();
      renderGridB();
    });

    el.clearBBtn.addEventListener('click', () => {
      state.matrixB = createMatrix(state.rowsB, state.colsB, () => 0);
      onMatrixDataChanged();
      renderGridB();
    });

    el.emptyBBtn.addEventListener('click', emptyMatrixB);

    // Mode Switcher (2 Matrices vs 3 Matrices)
    el.mode2MatricesBtn?.addEventListener('click', () => setMatrixCount(2));
    el.mode3MatricesBtn?.addEventListener('click', () => setMatrixCount(3));

    // Matrix Swap buttons (Top Toolbar & Stage Operator)
    el.swapABTopBtn?.addEventListener('click', swapMatrixAandB);
    el.swapStageBtn?.addEventListener('click', swapMatrixAandB);

    // Matrix C dimension inputs & steppers
    el.rowsC?.addEventListener('change', () => updateDimensionsC(el.rowsC.value, state.colsC));
    el.colsC?.addEventListener('change', () => updateDimensionsC(state.rowsC, el.colsC.value));
    el.decRowsC?.addEventListener('click', () => updateDimensionsC(state.rowsC - 1, state.colsC));
    el.incRowsC?.addEventListener('click', () => updateDimensionsC(state.rowsC + 1, state.colsC));
    el.decColsC?.addEventListener('click', () => updateDimensionsC(state.rowsC, state.colsC - 1));
    el.incColsC?.addEventListener('click', () => updateDimensionsC(state.rowsC, state.colsC + 1));

    // Matrix C quick actions
    el.emptyCBtn?.addEventListener('click', emptyMatrixC);
    el.randCBtn?.addEventListener('click', randMatrixC);
    el.clearCBtn?.addEventListener('click', clearMatrixC);

    // Batch Paste Modal
    el.pasteMatrixBtn.addEventListener('click', openPasteModal);
    el.closePasteModalBtn.addEventListener('click', closePasteModal);
    el.cancelPasteBtn.addEventListener('click', closePasteModal);
    el.applyPasteBtn.addEventListener('click', applyParsedPaste);

    el.tabMatrixA.addEventListener('click', () => {
      activePasteTarget = 'A';
      el.tabMatrixA.classList.add('active');
      el.tabMatrixB.classList.remove('active');
      el.tabMatrixCInput?.classList.remove('active');
      el.matrixRawInput.value = state.matrixA.map(r => r.join('\t')).join('\n');
    });

    el.tabMatrixB.addEventListener('click', () => {
      activePasteTarget = 'B';
      el.tabMatrixB.classList.add('active');
      el.tabMatrixA.classList.remove('active');
      el.tabMatrixCInput?.classList.remove('active');
      el.matrixRawInput.value = state.matrixB.map(r => r.join('\t')).join('\n');
    });

    if (el.tabMatrixCInput) {
      el.tabMatrixCInput.addEventListener('click', () => {
        activePasteTarget = 'C';
        el.tabMatrixCInput.classList.add('active');
        el.tabMatrixA.classList.remove('active');
        el.tabMatrixB.classList.remove('active');
        el.matrixRawInput.value = state.matrixCInput.map(r => r.join('\t')).join('\n');
      });
    }

    // Quick preset examples in Paste modal
    el.exampleIdentity.addEventListener('click', () => {
      el.matrixRawInput.value = '1 0 0\n0 1 0\n0 0 1';
    });
    el.exampleRotation.addEventListener('click', () => {
      // cos(45) ~ 0.707, -sin(45) ~ -0.707
      el.matrixRawInput.value = '0.71  -0.71\n0.71   0.71';
    });
    el.exampleFibonacci.addEventListener('click', () => {
      el.matrixRawInput.value = '1 1\n1 0';
    });

    // Export Modal
    el.exportBtn.addEventListener('click', openExportModal);
    el.closeExportModalBtn.addEventListener('click', closeExportModal);
    el.closeExportBtn.addEventListener('click', closeExportModal);

    const exportTabs = [
      { tab: el.tabExportLatex, format: 'latex' },
      { tab: el.tabExportJSON, format: 'json' },
      { tab: el.tabExportCSV, format: 'csv' },
      { tab: el.tabExportPython, format: 'python' }
    ];

    exportTabs.forEach(({ tab, format }) => {
      tab.addEventListener('click', () => {
        exportTabs.forEach(t => t.tab.classList.remove('active'));
        tab.classList.add('active');
        activeExportFormat = format;
        el.exportContentArea.value = generateExportString(format);
        el.copyStatusText.textContent = '';
      });
    });

    el.copyExportBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(el.exportContentArea.value);
        el.copyStatusText.textContent = '✓ Copied to clipboard successfully!';
        setTimeout(() => { el.copyStatusText.textContent = ''; }, 3000);
      } catch (err) {
        el.exportContentArea.select();
        document.execCommand('copy');
        el.copyStatusText.textContent = '✓ Copied to clipboard!';
      }
    });

    // Global Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Ignore if user is typing inside an input or textarea
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
        if (e.key === 'Escape' && state.activeModal) {
          closePasteModal();
          closeExportModal();
        }
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stopPlaying();
        stepForward();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stopPlaying();
        stepBackward();
      } else if (e.code === 'Home') {
        e.preventDefault();
        stopPlaying();
        jumpToFirst();
      } else if (e.code === 'End') {
        e.preventDefault();
        stopPlaying();
        jumpToLast();
      } else if (e.key === 'Escape' && state.activeModal) {
        closePasteModal();
        closeExportModal();
      }
    });
  }

  // Initialize application
  applyTheme(getPreferredTheme());
  setupEventListeners();
  initDefaultMatrices();
})();
