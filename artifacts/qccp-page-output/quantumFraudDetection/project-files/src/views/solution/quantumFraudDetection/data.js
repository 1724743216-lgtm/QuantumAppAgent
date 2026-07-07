/**
 * Mock data for quantum fraud detection page.
 * Used as fallback when API endpoints are unavailable.
 */

export const comparisonMetrics = {
  baseline: {
    method: 'Logistic Regression',
    f1_score: 0.4211,
    auc_roc: 0.695,
    precision: 0.3243,
    recall: 0.6,
    accuracy: 0.67,
  },
  quantum: {
    method: 'VQC',
    f1_score: 0.2476,
    auc_roc: 0.5723,
    precision: 0.2,
    recall: 0.325,
    accuracy: 0.605,
  },
};

export const algorithmParams = [
  { key: 'qubits', label: 'Qubits', value: 4 },
  { key: 'layers', label: 'Ansatz Layers', value: 2 },
  { key: 'encoding', label: 'Encoding', value: 'Angle Encoding' },
  { key: 'optimizer', label: 'Optimizer', value: 'Adam' },
  { key: 'learningRate', label: 'Learning Rate', value: 0.01 },
  { key: 'epochs', label: 'Training Epochs', value: 50 },
];

export const processSteps = [
  {
    step: 1,
    title: 'Data Preprocessing',
    description:
      'Raw transaction features are normalized and PCA-transformed to match quantum circuit input dimensions.',
  },
  {
    step: 2,
    title: 'Quantum Encoding',
    description:
      'Classical feature vectors are mapped onto quantum states via angle encoding using Ry rotations.',
  },
  {
    step: 3,
    title: 'Variational Circuit',
    description:
      'A parameterized ansatz with entangling CNOT gates and trainable Ry/Rz rotations processes the quantum state.',
  },
  {
    step: 4,
    title: 'Measurement',
    description:
      'Qubits are measured in the computational basis to produce classical bitstrings as output features.',
  },
  {
    step: 5,
    title: 'Classification',
    description:
      'Measured expectation values are fed into a classical post-processing layer for binary fraud/legitimate prediction.',
  },
];
