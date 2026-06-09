<template>
  <main class="terminal-shell">
    <aside class="agent-rail">
      <div class="brand-block">
        <div class="brand-mark">F</div>
        <div>
          <strong>FinAgent</strong>
          <span>Research Console</span>
        </div>
      </div>

      <nav class="agent-nav" aria-label="智能体导航">
        <button
          v-for="agent in agents"
          :key="agent.id"
          type="button"
          class="agent-button"
          :class="{ active: agent.id === activeAgentId }"
          @click="activeAgentId = agent.id"
        >
          <component :is="agent.icon" class="agent-icon" />
          <span>{{ agent.name }}</span>
          <small>{{ agent.short }}</small>
        </button>
      </nav>

      <div class="rail-foot">
        <span class="service-dot" :class="healthStatus"></span>
        <span>{{ healthText }}</span>
      </div>
    </aside>

    <section class="main-stage">
      <header class="top-bar">
        <div>
          <p class="eyebrow">Agentic RAG Console</p>
          <h1>{{ activeAgent.name }}</h1>
        </div>
        <div class="top-meta">
          <span>{{ activeAgent.endpoint }}</span>
          <el-button :icon="Refresh" circle title="检查服务状态" @click="refreshHealth" />
        </div>
      </header>

      <ConsultationWorkspace :agent="activeAgent" />
    </section>
  </main>
</template>

<script setup>
import { computed, markRaw, onMounted, ref } from 'vue'
import {
  Connection,
  DataAnalysis,
  DocumentChecked,
  Files,
  Refresh,
  Reading,
  TrendCharts,
} from '@element-plus/icons-vue'
import ConsultationWorkspace from './views/ConsultationWorkspace.vue'
import { checkHealth } from './api/consultations'

const agents = [
  {
    id: 'auto',
    name: '自动识别',
    short: 'Router',
    endpoint: '/api/agent/query',
    streamAgentId: 'auto',
    icon: markRaw(Connection),
    prompt: '请输入您的金融问题，系统将自动判断最合适的智能体。',
  },
  {
    id: 'advisory',
    name: '智能投顾',
    short: '组合与定投',
    endpoint: '/api/debug/investment-advisor',
    streamAgentId: 'advisory',
    icon: markRaw(TrendCharts),
    prompt: '我是稳健型投资者，计划用 30 万做三年期资产配置，希望兼顾流动性和回撤控制。',
  },
  {
    id: 'financial_report',
    name: '财报分析',
    short: '指标与质量',
    endpoint: '/api/debug/financial-report',
    streamAgentId: 'financial_report',
    icon: markRaw(Files),
    prompt: '请分析这份简化财报：营业收入同比增长 12%，净利润同比下降 8%，经营现金流为正但毛利率下滑。',
  },
  {
    id: 'risk_control',
    name: '风控审查',
    short: '暴露与缓释',
    endpoint: '/api/debug/risk-control',
    streamAgentId: 'risk_control',
    icon: markRaw(DataAnalysis),
    prompt: '审查一个组合：权益类 70%，债券类 10%，现金 20%，用户为中等风险承受能力。',
  },
  {
    id: 'compliance',
    name: '监管合规',
    short: '话术与边界',
    endpoint: '/api/debug/compliance',
    streamAgentId: 'compliance',
    icon: markRaw(DocumentChecked),
    prompt: '审查以下营销话术是否合规：本产品稳健增值，历史表现优秀，适合追求确定收益的客户。',
  },
  {
    id: 'education',
    name: '金融科普',
    short: '概念与防骗',
    endpoint: '/api/debug/education',
    streamAgentId: 'education',
    icon: markRaw(Reading),
    prompt: '请用适合新手的方式解释基金定投，并说明常见误区。',
  },
]

const activeAgentId = ref('advisory')
const healthStatus = ref('unknown')

const activeAgent = computed(() => agents.find((item) => item.id === activeAgentId.value) || agents[0])
const healthText = computed(() => {
  if (healthStatus.value === 'online') return 'Backend online'
  if (healthStatus.value === 'offline') return 'Backend offline'
  return 'Checking service'
})

async function refreshHealth() {
  healthStatus.value = 'unknown'
  try {
    await checkHealth()
    healthStatus.value = 'online'
  } catch {
    healthStatus.value = 'offline'
  }
}

onMounted(refreshHealth)
</script>

<style scoped>
.terminal-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 244px minmax(0, 1fr);
  background: #eef2ef;
  color: #17211c;
  font-family: Inter, "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
}

.agent-rail {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 18px 14px;
  background: #111a16;
  color: #edf3ef;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.brand-block {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 52px;
  padding: 4px 6px 18px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.brand-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border: 1px solid #b99d68;
  color: #f0d89e;
  font-weight: 700;
  background: #182620;
}

.brand-block strong,
.brand-block span {
  display: block;
}

.brand-block strong {
  font-size: 15px;
  letter-spacing: 0;
}

.brand-block span {
  margin-top: 3px;
  color: #8fa29a;
  font-size: 12px;
}

.agent-nav {
  display: grid;
  gap: 6px;
  margin-top: 18px;
}

.agent-button {
  height: 58px;
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr);
  grid-template-rows: 22px 18px;
  column-gap: 10px;
  align-items: center;
  border: 1px solid transparent;
  background: transparent;
  color: #dbe6df;
  text-align: left;
  cursor: pointer;
  padding: 8px 10px;
}

.agent-button:hover,
.agent-button.active {
  background: #1a2a23;
  border-color: #355245;
}

.agent-button.active {
  box-shadow: inset 3px 0 0 #c4a46a;
}

.agent-icon {
  grid-row: 1 / 3;
  width: 19px;
  height: 19px;
  color: #c4a46a;
}

.agent-button span {
  font-size: 14px;
  font-weight: 650;
  white-space: nowrap;
}

.agent-button small {
  color: #8fa29a;
  font-size: 12px;
}

.rail-foot {
  margin-top: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 38px;
  padding: 0 8px;
  color: #9caea6;
  font-size: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.service-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #9ca3af;
}

.service-dot.online {
  background: #2fa66a;
}

.service-dot.offline {
  background: #c86555;
}

.main-stage {
  min-width: 0;
  padding: 22px 24px 26px;
}

.top-bar {
  height: 62px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin: 0 auto 14px;
  max-width: 1520px;
}

.eyebrow {
  margin: 0 0 4px;
  color: #6d7f75;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

h1 {
  margin: 0;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: 0;
}

.top-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #66766d;
  font-size: 13px;
}

@media (max-width: 980px) {
  .terminal-shell {
    grid-template-columns: 1fr;
  }

  .agent-rail {
    min-height: auto;
    position: static;
  }

  .agent-nav {
    grid-template-columns: repeat(5, minmax(120px, 1fr));
    overflow-x: auto;
  }

  .rail-foot {
    display: none;
  }
}

@media (max-width: 680px) {
  .main-stage {
    padding: 16px 12px;
  }

  .top-bar {
    height: auto;
    flex-direction: column;
  }
}
</style>
