<script setup>
import { computed, onBeforeUnmount, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { ElMessage } from 'element-plus';
import Footer from '@/components/Footer.vue';
import { comparisonMetrics, algorithmParams, processSteps } from './data.js';

const { t } = useI18n();

// ==================== Comparison Metrics ====================
// Data loaded from data.js. Replace with API call via @/utils/axios.js
// when real backend endpoint and apiCode are available.
// See INTEGRATE.md for the API contract (GET /api/comparison).
const metricsData = ref(comparisonMetrics);

const metricRows = computed(() => {
  if (!metricsData.value) return [];
  const { baseline, quantum } = metricsData.value;
  const fields = ['f1_score', 'auc_roc', 'precision', 'recall', 'accuracy'];
  return fields.map((field) => ({
    metric: field,
    baselineValue: baseline[field],
    quantumValue: quantum[field],
  }));
});

// ==================== Prediction Form ====================
const predictFormRef = ref(null);
const predictLoading = ref(false);
const predictResult = ref(null);

const predictForm = reactive({
  amount: '',
  timeGap: '',
  freqRatio: '',
  riskScore: '',
});

const predictRules = computed(() => ({
  amount: [
    { required: true, message: t('quantumFraudDetection.form.validation.amountRequired'), trigger: 'blur' },
    { pattern: /^-?\d+(\.\d+)?$/, message: t('quantumFraudDetection.form.validation.numberInvalid'), trigger: 'blur' },
  ],
  timeGap: [
    { required: true, message: t('quantumFraudDetection.form.validation.timeGapRequired'), trigger: 'blur' },
    { pattern: /^-?\d+(\.\d+)?$/, message: t('quantumFraudDetection.form.validation.numberInvalid'), trigger: 'blur' },
  ],
  freqRatio: [
    { required: true, message: t('quantumFraudDetection.form.validation.freqRatioRequired'), trigger: 'blur' },
    { pattern: /^-?\d+(\.\d+)?$/, message: t('quantumFraudDetection.form.validation.numberInvalid'), trigger: 'blur' },
  ],
  riskScore: [
    { required: true, message: t('quantumFraudDetection.form.validation.riskScoreRequired'), trigger: 'blur' },
    { pattern: /^-?\d+(\.\d+)?$/, message: t('quantumFraudDetection.form.validation.numberInvalid'), trigger: 'blur' },
  ],
}));

const resetForm = () => {
  predictResult.value = null;
  predictFormRef.value?.resetFields();
};

const submitPrediction = async () => {
  const valid = await predictFormRef.value?.validate().catch(() => false);
  if (!valid) return;

  predictLoading.value = true;
  predictResult.value = null;

  const payload = {
    amount: Number(predictForm.amount),
    timeGap: Number(predictForm.timeGap),
    freqRatio: Number(predictForm.freqRatio),
    riskScore: Number(predictForm.riskScore),
  };

  // Simulate prediction locally from data.js.
  // Replace with API call via @/utils/axios.js when backend is ready.
  // See INTEGRATE.md for the API contract (POST /api/predict).
  setTimeout(() => {
    const isFraud = payload.riskScore > 0.5;
    predictResult.value = {
      label: isFraud ? 1 : 0,
      probability: isFraud
        ? (0.6 + Math.random() * 0.3).toFixed(4)
        : (0.1 + Math.random() * 0.3).toFixed(4),
    };
    ElMessage.info(t('quantumFraudDetection.message.mockPrediction'));
    predictLoading.value = false;
  }, 600);
};

// ==================== Metric label helper ====================
const metricLabelMap = {
  f1_score: 'quantumFraudDetection.metrics.f1Score',
  auc_roc: 'quantumFraudDetection.metrics.aucRoc',
  precision: 'quantumFraudDetection.metrics.precision',
  recall: 'quantumFraudDetection.metrics.recall',
  accuracy: 'quantumFraudDetection.metrics.accuracy',
};

const getMetricLabel = (key) => {
  return t(metricLabelMap[key] || key);
};

const formatValue = (val) => {
  if (typeof val === 'number') {
    return val.toFixed(4);
  }
  return val;
};

// ==================== Lifecycle ====================
// No onMounted needed: data is initialized from data.js synchronously.
// When API is integrated, replace with async load via @/utils/axios.js.

onBeforeUnmount(() => {
  // cleanup timers or listeners if any
});
</script>

<template>
  <main class="quantum-fraud-detection-page">
    <!-- ==================== Banner Section ==================== -->
    <section class="banner-section">
      <div class="banner-content wrapper">
        <h1 class="banner-title">{{ $t('quantumFraudDetection.banner.title') }}</h1>
        <p class="banner-subtitle">{{ $t('quantumFraudDetection.banner.subtitle') }}</p>
      </div>
    </section>

    <!-- ==================== Algorithm Architecture Section ==================== -->
    <section class="architecture-section wrapper">
      <h2 class="section-title">{{ $t('quantumFraudDetection.architecture.title') }}</h2>
      <el-steps :active="5" finish-status="success" align-center class="process-steps">
        <el-step
          v-for="step in processSteps"
          :key="step.step"
          :title="$t(`quantumFraudDetection.architecture.step${step.step}.title`)"
          :description="$t(`quantumFraudDetection.architecture.step${step.step}.desc`)"
        />
      </el-steps>
      <div class="step-cards">
        <div
          v-for="step in processSteps"
          :key="step.step"
          class="step-card"
        >
          <div class="step-card-number">{{ step.step }}</div>
          <div class="step-card-title">{{ $t(`quantumFraudDetection.architecture.step${step.step}.title`) }}</div>
          <div class="step-card-desc">{{ $t(`quantumFraudDetection.architecture.step${step.step}.desc`) }}</div>
        </div>
      </div>
    </section>

    <!-- ==================== Metrics Comparison Section ==================== -->
    <section class="metrics-section wrapper">
      <h2 class="section-title">{{ $t('quantumFraudDetection.metrics.title') }}</h2>

      <el-table
        :data="metricRows"
        border
        class="comparison-table"
        style="width: 100%"
      >
        <el-table-column
          :label="$t('quantumFraudDetection.metrics.metricCol')"
          prop="metric"
          min-width="160"
        >
          <template #default="{ row }">
            {{ getMetricLabel(row.metric) }}
          </template>
        </el-table-column>
        <el-table-column
          :label="$t('quantumFraudDetection.metrics.baselineCol')"
          min-width="180"
        >
          <template #default="{ row }">
            <span class="metric-value baseline">{{ formatValue(row.baselineValue) }}</span>
          </template>
        </el-table-column>
        <el-table-column
          :label="$t('quantumFraudDetection.metrics.quantumCol')"
          min-width="180"
        >
          <template #default="{ row }">
            <span class="metric-value quantum">{{ formatValue(row.quantumValue) }}</span>
          </template>
        </el-table-column>
      </el-table>

      <div class="radar-note">
        <p class="radar-note-text">{{ $t('quantumFraudDetection.metrics.radarNote') }}</p>
      </div>
    </section>

    <!-- ==================== Online Detection Section ==================== -->
    <section class="detection-section wrapper">
      <h2 class="section-title">{{ $t('quantumFraudDetection.detection.title') }}</h2>

      <div class="detection-container">
        <el-form
          ref="predictFormRef"
          :model="predictForm"
          :rules="predictRules"
          label-position="top"
          class="detection-form"
        >
          <el-form-item
            :label="$t('quantumFraudDetection.form.amountLabel')"
            prop="amount"
          >
            <el-input
              v-model="predictForm.amount"
              :placeholder="$t('quantumFraudDetection.form.amountPlaceholder')"
            />
          </el-form-item>

          <el-form-item
            :label="$t('quantumFraudDetection.form.timeGapLabel')"
            prop="timeGap"
          >
            <el-input
              v-model="predictForm.timeGap"
              :placeholder="$t('quantumFraudDetection.form.timeGapPlaceholder')"
            />
          </el-form-item>

          <el-form-item
            :label="$t('quantumFraudDetection.form.freqRatioLabel')"
            prop="freqRatio"
          >
            <el-input
              v-model="predictForm.freqRatio"
              :placeholder="$t('quantumFraudDetection.form.freqRatioPlaceholder')"
            />
          </el-form-item>

          <el-form-item
            :label="$t('quantumFraudDetection.form.riskScoreLabel')"
            prop="riskScore"
          >
            <el-input
              v-model="predictForm.riskScore"
              :placeholder="$t('quantumFraudDetection.form.riskScorePlaceholder')"
            />
          </el-form-item>

          <div class="form-actions">
            <el-button
              type="primary"
              :loading="predictLoading"
              @click="submitPrediction"
            >
              {{ $t('quantumFraudDetection.actions.predict') }}
            </el-button>
            <el-button @click="resetForm">
              {{ $t('quantumFraudDetection.actions.reset') }}
            </el-button>
          </div>
        </el-form>

        <div v-if="predictResult" class="prediction-result">
          <h3 class="result-title">{{ $t('quantumFraudDetection.result.title') }}</h3>
          <div class="result-item">
            <span class="result-label">{{ $t('quantumFraudDetection.result.label') }}</span>
            <el-tag
              :type="predictResult.label === 1 ? 'danger' : 'success'"
              class="result-tag"
            >
              {{ predictResult.label === 1
                ? $t('quantumFraudDetection.result.fraud')
                : $t('quantumFraudDetection.result.legitimate')
              }}
            </el-tag>
          </div>
          <div class="result-item">
            <span class="result-label">{{ $t('quantumFraudDetection.result.probability') }}</span>
            <span class="result-value">{{ predictResult.probability }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- ==================== Technical Parameters Section ==================== -->
    <section class="params-section wrapper">
      <h2 class="section-title">{{ $t('quantumFraudDetection.params.title') }}</h2>
      <div class="params-grid">
        <div
          v-for="param in algorithmParams"
          :key="param.key"
          class="param-card"
        >
          <div class="param-label">{{ $t(`quantumFraudDetection.params.${param.key}`) }}</div>
          <div class="param-value">{{ param.value }}</div>
        </div>
      </div>
    </section>

    <Footer />
  </main>
</template>

<style lang="scss" scoped>
.quantum-fraud-detection-page {
  width: 100%;
  min-height: calc(100vh - 60px);
  background: #f4f7fc;
}

// ==================== Tokens ====================
$primary: #1664ff;
$text-title: #020814;
$text-sub: #41464f;
$text-body: #939aab;
$page-bg: #f4f7fc;
$card-bg: #ffffff;
$border-line: #dce0eb;
$radius-card: 8px;
$radius-btn: 4px;
$radius-tag: 6px;

// ==================== Shared ====================
.wrapper {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 140px;

  @media (max-width: 1440px) {
    padding: 0 60px;
  }

  @media (max-width: 1366px) {
    padding: 0 40px;
  }
}

.section-title {
  font-size: 30px;
  font-weight: bold;
  color: $text-title;
  margin-bottom: 30px;
}

// ==================== Banner ====================
.banner-section {
  background: linear-gradient(135deg, #0a1a3f 0%, #122866 50%, #1664ff 100%);
  padding: 80px 0;

  .banner-content {
    text-align: center;
  }

  .banner-title {
    font-size: 60px;
    font-weight: bold;
    color: #ffffff;
    margin: 0 0 16px;
    line-height: 1.2;
  }

  .banner-subtitle {
    font-size: 20px;
    color: rgba(255, 255, 255, 0.8);
    margin: 0;
  }
}

// ==================== Architecture ====================
.architecture-section {
  padding: 60px 0;

  .process-steps {
    margin-bottom: 40px;
  }

  .step-cards {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 20px;
  }

  .step-card {
    background: $card-bg;
    border: 1px solid $border-line;
    border-radius: $radius-card;
    padding: 24px 20px;
    text-align: center;
    transition: border-color 0.2s;

    &:hover {
      border-color: $primary;
    }
  }

  .step-card-number {
    width: 36px;
    height: 36px;
    line-height: 36px;
    border-radius: 50%;
    background: $primary;
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
    margin: 0 auto 12px;
  }

  .step-card-title {
    font-size: 18px;
    font-weight: bold;
    color: $text-title;
    margin-bottom: 8px;
  }

  .step-card-desc {
    font-size: 14px;
    color: $text-body;
    line-height: 1.6;
  }
}

// ==================== Metrics ====================
.metrics-section {
  padding: 60px 0;

  .comparison-table {
    border-radius: $radius-card;
    overflow: hidden;
  }

  .metric-value {
    font-size: 18px;
    font-weight: bold;

    &.baseline {
      color: $text-sub;
    }

    &.quantum {
      color: $primary;
    }
  }

  .radar-note {
    margin-top: 20px;
    background: $card-bg;
    border: 1px solid $border-line;
    border-radius: $radius-card;
    padding: 20px;
  }

  .radar-note-text {
    font-size: 14px;
    color: $text-body;
    margin: 0;
    line-height: 1.6;
  }
}

// ==================== Detection ====================
.detection-section {
  padding: 60px 0;

  .detection-container {
    background: $card-bg;
    border: 1px solid $border-line;
    border-radius: $radius-card;
    padding: 40px;
    display: flex;
    gap: 40px;

    @media (max-width: 1366px) {
      flex-direction: column;
      gap: 24px;
    }
  }

  .detection-form {
    flex: 1;
    min-width: 0;
  }

  .form-actions {
    display: flex;
    gap: 12px;
    margin-top: 8px;

    .el-button--primary {
      border-radius: $radius-btn;
    }

    .el-button--default {
      border-radius: $radius-btn;
    }
  }

  .prediction-result {
    flex: 0 0 320px;
    background: #f3f7ff;
    border: 1px solid $border-line;
    border-radius: $radius-card;
    padding: 32px;

    @media (max-width: 1366px) {
      flex: none;
    }
  }

  .result-title {
    font-size: 24px;
    font-weight: bold;
    color: $text-title;
    margin: 0 0 20px;
  }

  .result-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .result-label {
    font-size: 16px;
    color: $text-sub;
  }

  .result-value {
    font-size: 20px;
    font-weight: bold;
    color: $primary;
  }

  .result-tag {
    border-radius: $radius-tag;
  }
}

// ==================== Parameters ====================
.params-section {
  padding: 60px 0 80px;

  .params-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
  }

  .param-card {
    background: $card-bg;
    border: 1px solid $border-line;
    border-radius: $radius-card;
    padding: 24px;
    text-align: center;
    transition: border-color 0.2s;

    &:hover {
      border-color: $primary;
    }
  }

  .param-label {
    font-size: 14px;
    color: $text-body;
    margin-bottom: 8px;
  }

  .param-value {
    font-size: 24px;
    font-weight: bold;
    color: $text-title;
  }
}
</style>
