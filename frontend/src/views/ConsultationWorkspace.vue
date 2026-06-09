<template>
  <div class="console-grid">
    <section class="command-panel">
      <div class="panel-head">
        <span>Task Brief</span>
        <el-tag effect="plain" type="info">{{ agent.id }}</el-tag>
      </div>

      <el-input
        v-model="question"
        type="textarea"
        resize="none"
        :rows="8"
        maxlength="2400"
        show-word-limit
        placeholder="输入需要审查、分析或解释的金融问题"
      />

      <div class="quick-row">
        <button type="button" @click="useAgentPrompt">载入样例</button>
        <button type="button" @click="clearResult">清空结果</button>
      </div>

      <div class="field-block">
        <label>用户画像</label>
        <div class="profile-grid">
          <el-select v-model="profile.risk_preference" placeholder="风险偏好" clearable>
            <el-option label="保守型" value="low" />
            <el-option label="稳健型" value="stable" />
            <el-option label="平衡型" value="balanced" />
            <el-option label="进取型" value="aggressive" />
          </el-select>
          <el-select v-model="profile.investment_experience" placeholder="经验" clearable>
            <el-option label="新手" value="beginner" />
            <el-option label="有经验" value="experienced" />
          </el-select>
          <el-select v-model="profile.income_level" placeholder="收入水平" clearable>
            <el-option label="较低" value="low" />
            <el-option label="中等" value="medium" />
            <el-option label="较高" value="high" />
          </el-select>
          <el-select v-model="profile.liquidity_need" placeholder="流动性" clearable>
            <el-option label="低" value="low" />
            <el-option label="中" value="medium" />
            <el-option label="高" value="high" />
          </el-select>
        </div>
      </div>

      <div class="field-block">
        <label>结构化补充</label>
        <el-input
          v-model="extraContext"
          type="textarea"
          resize="none"
          :rows="4"
          placeholder="可选。持仓、财报摘要、待审文案、风险条件等"
        />
      </div>

      <div class="submit-row">
        <el-button :icon="Connection" @click="showSample">预览样式</el-button>
        <el-button type="primary" :loading="loading" @click="submit">运行智能体</el-button>
      </div>

      <el-alert
        v-if="slowHint"
        class="notice"
        type="warning"
        :title="slowHint"
        show-icon
        :closable="false"
      />
      <el-alert
        v-if="error"
        class="notice"
        type="error"
        :title="error"
        show-icon
        :closable="false"
      />
    </section>

    <section class="analysis-stage">
      <div class="process-strip">
        <div
          v-for="step in processSteps"
          :key="step.key"
          class="process-step"
          :class="step.state"
        >
          <span></span>
          <p>{{ step.label }}</p>
        </div>
      </div>

      <el-alert
        v-if="routerInfo && agent.id === 'auto'"
        class="router-banner"
        type="success"
        :title="`已自动路由至：${intentLabel(routerInfo.selected_intent)}  (置信度 ${(routerInfo.confidence * 100).toFixed(0)}%)`"
        :closable="false"
        show-icon
      />

      <ResultPanel
        :agent="agent"
        :result="displayResult"
        :debug="debugMeta"
        :loading="loading"
      />

      <ReasoningTimeline
        :events="timelineEvents"
      />
    </section>
  </div>
</template>

<script setup>
import { computed, onUnmounted, reactive, ref, watch } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import ResultPanel from '../components/ResultPanel.vue'
import ReasoningTimeline from '../components/ReasoningTimeline.vue'
import {
  postDebugAgent,
  prepareStreamAgent,
  openAgentStream,
} from '../api/consultations'

const props = defineProps({
  agent: {
    type: Object,
    required: true,
  },
})

const question = ref(props.agent.prompt)
const extraContext = ref('')
const loading = ref(false)
const loadingStatus = ref('')     // 'connecting' | 'waiting' | 'timeout' | 'cancelled' | 'failed' | ''
const error = ref('')
const result = ref(null)
const debugMeta = ref(makeSampleDebug(props.agent))

// 请求隔离：避免切换智能体后，旧请求返回覆盖新页面。
const abortController = ref(null)
const currentRequestId = ref(0)
const slowTimer = ref(null)
const SLOW_HINT_MS = 20000  // Show slow hint after 20s

// SSE 状态：stream 负责实时推理链，HTTP 仅作为降级路径。
const timelineEvents = ref([])
const streamStatus = ref('idle')  // idle | preparing | streaming | done | error
const activeRunId = ref('')
const eventSource = ref(null)
const routerInfo = ref(null)    // RouterDecision from auto-routing

const profile = reactive({
  risk_preference: 'stable',
  investment_experience: 'experienced',
  income_level: '',
  liquidity_need: 'medium',
})

const sampleResult = computed(() => makeSampleResult(props.agent))
const displayResult = computed(() => result.value || sampleResult.value)

const processSteps = computed(() => {
  if (loading.value) {
    const statusLabel = loadingStatus.value === 'waiting'
      ? '等待 DeepAgent 多轮调用工具…'
      : loadingStatus.value === 'connecting'
      ? '连接后端…'
      : '处理中…'
    return [
      { key: 'intent', label: '任务已提交', state: 'done' },
      { key: 'retrieval', label: statusLabel, state: 'active' },
      { key: 'tools', label: '等待调用工具', state: 'idle' },
      { key: 'report', label: '等待生成报告', state: 'idle' },
    ]
  }
  if (loadingStatus.value === 'cancelled') {
    return [
      { key: 'intent', label: '请求已取消', state: 'done' },
      { key: 'retrieval', label: '已切换智能体', state: 'idle' },
      { key: 'tools', label: '等待运行', state: 'idle' },
      { key: 'report', label: '样式预览', state: 'active' },
    ]
  }
  if (loadingStatus.value === 'timeout') {
    return [
      { key: 'intent', label: '请求超时', state: 'done' },
      { key: 'retrieval', label: '后端可能仍在处理', state: 'idle' },
      { key: 'tools', label: '等待运行', state: 'idle' },
      { key: 'report', label: '样式预览', state: 'active' },
    ]
  }
  const hasRealResult = Boolean(result.value)
  return [
    { key: 'intent', label: loadingStatus.value === 'failed' ? '请求失败' : '任务就绪', state: loadingStatus.value === 'failed' ? 'done' : 'done' },
    { key: 'retrieval', label: hasRealResult ? '证据已回填' : '证据预览', state: hasRealResult ? 'done' : 'idle' },
    { key: 'tools', label: hasRealResult ? '工具轨迹' : '等待运行', state: hasRealResult ? 'done' : 'idle' },
    { key: 'report', label: hasRealResult ? '报告完成' : '样式预览', state: hasRealResult ? 'done' : 'active' },
  ]
})

const slowHint = computed(() => {
  if (!loading.value) return ''
  if (loadingStatus.value === 'waiting') {
    return 'DeepAgent 正在多轮调用工具，可能需要 30–120 秒，请耐心等待…'
  }
  if (loadingStatus.value === 'connecting') {
    return '正在连接后端服务…'
  }
  return ''
})

watch(
  () => props.agent.id,
  () => {
    // Cancel any in-flight request for the PREVIOUS agent
    cancelCurrentRequest()
    question.value = props.agent.prompt
    result.value = null
    debugMeta.value = makeSampleDebug(props.agent)
    error.value = ''
    loadingStatus.value = ''
  }
)

function compactProfile() {
  const clean = {}
  Object.entries(profile).forEach(([key, value]) => {
    if (value) clean[key] = value
  })
  if (extraContext.value.trim()) {
    clean.extra_context = extraContext.value.trim()
  }
  return clean
}

function useAgentPrompt() {
  question.value = props.agent.prompt
}

function clearResult() {
  cancelCurrentRequest()
  result.value = null
  debugMeta.value = makeSampleDebug(props.agent)
  error.value = ''
  loadingStatus.value = ''
  routerInfo.value = null
}

function showSample() {
  cancelCurrentRequest()
  result.value = null
  debugMeta.value = makeSampleDebug(props.agent)
  error.value = ''
  loadingStatus.value = ''
  routerInfo.value = null
}

/** Map intent code to Chinese display name. */
function intentLabel(intent) {
  const map = {
    advisory: '智能投顾',
    financial_report: '财报分析',
    risk_control: '风控审查',
    compliance: '监管合规',
    education: '金融科普',
  }
  return map[intent] || intent
}

function cancelCurrentRequest() {
  if (abortController.value) {
    abortController.value.abort()
    abortController.value = null
  }
  if (slowTimer.value) {
    clearTimeout(slowTimer.value)
    slowTimer.value = null
  }
  // Close SSE stream
  if (eventSource.value) {
    closeEventSource()
  }
  routerInfo.value = null
}

function closeEventSource() {
  if (eventSource.value) {
    eventSource.value.events.close()
    eventSource.value = null
  }
  streamStatus.value = 'idle'
  activeRunId.value = ''
}

async function submit() {
  if (!question.value.trim()) {
    error.value = '请输入任务内容。'
    return
  }

  // Cancel any previous in-flight request or stream
  cancelCurrentRequest()

  error.value = ''
  loading.value = true
  loadingStatus.value = 'connecting'
  timelineEvents.value = []
  routerInfo.value = null

  // ── Concurrency: request ID for anti-stale ─────────────────────
  const thisRequestId = ++currentRequestId.value

  // Use streamAgentId (for SSE) with fallback to agent.id (for regular HTTP)
  const sseAgentId = props.agent.streamAgentId || props.agent.id

  // ── Try SSE streaming first ────────────────────────────────────
  try {
    streamStatus.value = 'preparing'
    const prepData = await prepareStreamAgent(sseAgentId, {
      question: question.value.trim(),
      user_profile: compactProfile(),
    })

    if (thisRequestId !== currentRequestId.value) return

    activeRunId.value = prepData.run_id
    streamStatus.value = 'streaming'

    // Open SSE connection
    const es = openAgentStream(prepData.stream_url)
    eventSource.value = es

    es.events.addEventListener('run_started', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      appendTimelineEvent(JSON.parse(e.data), 'done')
    })

    es.events.addEventListener('intent_selected', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      const parsed = JSON.parse(e.data)
      if (parsed.router) {
        routerInfo.value = parsed.router
      }
      appendTimelineEvent(parsed, 'done')
    })

    es.events.addEventListener('retrieval_started', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      appendTimelineEvent(JSON.parse(e.data), 'active')
    })

    es.events.addEventListener('retrieval_done', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      const parsed = JSON.parse(e.data)
      appendTimelineEvent(parsed, 'done')
    })

    es.events.addEventListener('agent_started', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      appendTimelineEvent(JSON.parse(e.data), 'active')
      loadingStatus.value = 'waiting'
    })

    es.events.addEventListener('heartbeat', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      const parsed = JSON.parse(e.data)
      loadingStatus.value = 'waiting'
      // Update the active event's label without adding a new one
      if (timelineEvents.value.length > 0) {
        const last = timelineEvents.value[timelineEvents.value.length - 1]
        if (last._state === 'active') {
          last.label = parsed.label || last.label
          last.message = parsed.message || last.message
        }
      }
    })

    // ── Real-time tool events (tool_started → tool_done/tool_failed) ──
    es.events.addEventListener('tool_started', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      const parsed = JSON.parse(e.data)
      appendTimelineEvent({ ...parsed, _toolState: 'active' }, 'active')
      loadingStatus.value = 'waiting'
    })

    es.events.addEventListener('tool_done', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastToolActiveDone()
      appendTimelineEvent(JSON.parse(e.data), 'done')
    })

    es.events.addEventListener('tool_failed', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastToolActiveDone()
      appendTimelineEvent(JSON.parse(e.data), 'error')
    })

    // ── Legacy tool_call events (pipeline fallback) ─────────────
    es.events.addEventListener('tool_call', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      appendTimelineEvent(JSON.parse(e.data), 'done')
    })

    es.events.addEventListener('fallback_used', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      appendTimelineEvent(JSON.parse(e.data), 'done')
    })

    es.events.addEventListener('report_done', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      appendTimelineEvent(JSON.parse(e.data), 'done')
    })

    es.events.addEventListener('run_done', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      markLastEventDone()
      const parsed = JSON.parse(e.data)
      appendTimelineEvent(parsed, 'done')

      // Store router info (auto-routing)
      if (parsed.router) {
        routerInfo.value = parsed.router
      }

      // Populate result from final event
      if (parsed.response) {
        result.value = parsed.response
      }
      if (parsed.agent_architecture || parsed.evidence) {
        debugMeta.value = normalizeDebug({
          agent_architecture: parsed.agent_architecture || {},
          evidence: parsed.evidence || {},
          planned_queries: parsed.planned_queries || [],
          router: parsed.router || null,
        }, props.agent)
      }
      loading.value = false
      loadingStatus.value = ''
      streamStatus.value = 'done'
      closeEventSource()
    })

    es.events.addEventListener('run_error', (e) => {
      if (thisRequestId !== currentRequestId.value) return
      const parsed = JSON.parse(e.data)
      markLastEventDone()
      appendTimelineEvent(parsed, 'error')
      error.value = parsed.message || '流式运行失败'
      loading.value = false
      loadingStatus.value = 'failed'
      streamStatus.value = 'error'
      closeEventSource()
    })

    // Error handler for connection issues
    es.events.onerror = () => {
      if (thisRequestId !== currentRequestId.value) return
      if (streamStatus.value === 'done' || streamStatus.value === 'error') {
        closeEventSource()
        return
      }
      // Connection lost mid-stream — fall back to HTTP
      closeEventSource()
      submitFallbackHttp(thisRequestId)
    }

    // ── Slow hint timer ───────────────────────────────────────
    slowTimer.value = setTimeout(() => {
      if (loading.value && thisRequestId === currentRequestId.value) {
        loadingStatus.value = 'waiting'
      }
    }, SLOW_HINT_MS)

  } catch (err) {
    // SSE prepare failed — fall back to HTTP
    if (thisRequestId !== currentRequestId.value) return

    if (err.name === 'AbortError') {
      loadingStatus.value = 'cancelled'
      loading.value = false
      return
    }
    // Fallback to regular HTTP POST
    submitFallbackHttp(thisRequestId)
  }
}

/** Fallback: regular HTTP POST when SSE prepare fails or stream disconnects. */
async function submitFallbackHttp(thisRequestId) {
  if (thisRequestId !== currentRequestId.value) return

  // Reset timeline
  timelineEvents.value = []
  appendTimelineEvent({
    label: 'HTTP 模式运行', message: 'SSE 不可用，回退到普通请求', type: 'run_started',
  }, 'active')
  loadingStatus.value = 'waiting'
  routerInfo.value = null

  // Use agent query endpoint for auto, debug endpoint for manual
  const fallbackEndpoint = props.agent.id === 'auto'
    ? '/api/agent/query'
    : props.agent.endpoint

  const controller = new AbortController()
  abortController.value = controller

  slowTimer.value = setTimeout(() => {
    if (loading.value && thisRequestId === currentRequestId.value) {
      loadingStatus.value = 'waiting'
    }
  }, SLOW_HINT_MS)

  try {
    const data = await postDebugAgent(
      fallbackEndpoint,
      {
        question: question.value.trim(),
        user_profile: compactProfile(),
      },
      { signal: controller.signal }
    )

    if (thisRequestId !== currentRequestId.value) return

    markLastEventDone()
    appendTimelineEvent({ label: '请求完成', message: '报告已生成', type: 'report_done' }, 'done')
    // For auto-routing, the response is flat (not nested under .response)
    result.value = data.response || data
    if (data.router) {
      routerInfo.value = data.router
    }
    debugMeta.value = normalizeDebug(data, props.agent)
    loadingStatus.value = ''
    error.value = ''
  } catch (err) {
    if (thisRequestId !== currentRequestId.value) return
    if (err.name === 'AbortError') {
      loadingStatus.value = 'cancelled'
    } else if (err.name === 'TimeoutError') {
      loadingStatus.value = 'timeout'
      error.value = err.message
      markLastEventDone()
      appendTimelineEvent({ label: '请求超时', message: err.message, type: 'run_error' }, 'error')
    } else {
      loadingStatus.value = 'failed'
      error.value = err.message || '请求失败，请检查后端服务。'
      markLastEventDone()
      appendTimelineEvent({ label: '请求失败', message: error.value, type: 'run_error' }, 'error')
    }
  } finally {
    if (thisRequestId === currentRequestId.value) {
      loading.value = false
      abortController.value = null
      if (slowTimer.value) {
        clearTimeout(slowTimer.value)
        slowTimer.value = null
      }
    }
  }
}

// ── Timeline helpers ────────────────────────────────────────────

function appendTimelineEvent(data, state) {
  timelineEvents.value.push({
    ...data,
    _state: state,
    _time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
  })
}

function markLastEventDone() {
  const events = timelineEvents.value
  if (events.length > 0 && events[events.length - 1]._state === 'active') {
    events[events.length - 1]._state = 'done'
  }
}

/** Mark the last tool_started event as done (matched by _toolState). */
function markLastToolActiveDone() {
  const events = timelineEvents.value
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]._toolState === 'active') {
      events[i]._state = 'done'
      break
    }
  }
}

onUnmounted(() => {
  cancelCurrentRequest()
})

function normalizeDebug(data, agent) {
  const architecture = data.agent_architecture || data.response?.agent_architecture || {}
  return {
    endpoint: agent.endpoint,
    mode: data.mode || 'debug',
    actual_architecture: architecture.agent_architecture || architecture.actual_architecture || data.actual_architecture || 'deepagent',
    fallback_used: architecture.fallback_used ?? data.fallback_used ?? false,
    tool_count: architecture.tool_count ?? data.tool_count ?? 0,
    tool_traces: architecture.tool_traces || data.tool_traces || [],
    evidence: data.evidence || {},
    planned_queries: data.planned_queries || [],
  }
}

function makeSampleDebug(agent) {
  return {
    endpoint: agent.endpoint,
    mode: 'preview',
    actual_architecture: 'deepagent',
    fallback_used: false,
    tool_count: agent.id === 'education' ? 1 : 7,
    tool_traces: sampleTraces(agent),
    evidence: { item_count: 4, has_compliance: true, has_risk: true },
    planned_queries: [
      { collection: 'advisory_knowledge', query: '风险偏好 资产配置 证据' },
      { collection: 'compliance_knowledge', query: '金融服务 合规边界 风险提示' },
    ],
  }
}

function makeSampleResult(agent) {
  const samples = {
    advisory: {
      title: '稳健组合配置简报',
      answer:
        '一、用户画像摘要\n用户风险偏好偏稳健，目标期限为三年，关注流动性与回撤控制。\n\n二、配置思路\n建议以低波动资产作为底仓，权益类资产控制在可承受范围内，现金及货币类资产保留应急空间。\n\n三、组合建议\n现金及货币类 15%-25%；中短债及固收类 35%-45%；宽基指数基金类 20%-30%；黄金或商品类 5%-10%。\n\n四、风险提示\n以上为资产类别层面的配置建议，不涉及个股、基金代码或收益承诺。',
    },
    financial_report: {
      title: '经营质量观察',
      answer:
        '一、财报对象与数据范围\n基于用户输入的简化财务摘要进行结构化分析。\n\n二、核心指标摘要\n收入增长 12%，净利润下降 8%，经营现金流为正，毛利率下滑。\n\n三、初步判断\n收入扩张与盈利承压并存，需重点核查成本端、费用端和产品结构变化。\n\n四、风险提示\n该分析不构成投资建议，完整判断需要结合资产负债表、现金流量表与附注。',
    },
    risk_control: {
      title: '组合风险审查',
      answer:
        '一、审查对象与输入范围\n组合权益类占比 70%，债券类 10%，现金 20%。\n\n二、主要风险暴露\n权益类占比较高，与中等风险承受能力存在一定错配。\n\n三、缓释建议\n可通过降低权益仓位、增加固收与现金缓冲、设置再平衡阈值来控制波动。\n\n四、风险提示\n本报告仅供风险管理参考，不构成投资建议。',
    },
    compliance: {
      title: '营销话术合规审查',
      answer:
        '一、审查对象与场景说明\n审查对象为金融产品营销话术。\n\n二、合规风险等级\n中等风险。\n\n三、违规或高风险表述识别\n“确定收益”存在被理解为收益承诺的风险。\n\n四、整改建议\n建议改为强调风险揭示、适当性匹配和过往业绩不代表未来表现。\n\n八、合规提示\n本报告仅供合规风险识别参考，不构成正式法律意见，请审慎判断。',
    },
    education: {
      title: '基金定投科普',
      answer:
        '一、概念解释\n基金定投是按固定频率投入固定或近似固定金额的投资方式。\n\n二、适用场景\n更适合长期、分批、纪律化参与市场，不适合期待短期确定收益。\n\n三、常见误区\n定投不等于稳赚，也不能替代风险测评。\n\n四、学习建议\n先理解波动、净值、费率和持有期限，再讨论具体产品类别。',
    },
  }

  const selected = samples[agent.id] || samples.advisory
  return {
    intent: agent.id,
    agent: `${agent.id}_deepagent`,
    answer: selected.answer,
    title: selected.title,
    sources: [
      { title: '投资者适当性管理要求', source_type: 'compliance', confidence: 0.86 },
      { title: '资产配置基础模型说明', source_type: 'advisory', confidence: 0.81 },
      { title: '风险揭示与教育材料', source_type: 'risk', confidence: 0.78 },
    ],
    warnings: [],
    risk_notice: '输出仅供研究和风险识别参考，不构成投资建议或正式法律意见。',
  }
}

function sampleTraces(agent) {
  const map = {
    advisory: [['profile_analyzer', '用户画像解析器'], ['risk_assessor', '风险评估器'], ['allocation_engine', '资产配置引擎']],
    financial_report: [['metric_extractor', '财务指标提取器'], ['quality_checker', '经营质量分析器']],
    risk_control: [['risk_exposure_checker', '风险暴露识别器'], ['mitigation_planner', '缓释建议规划器']],
    compliance: [['prohibited_expression_detector', '违规话术检测器'], ['compliance_output_policy', '合规输出审查器']],
    education: [['simple_concept_workflow', '简单概念投教工作流']],
  }
  return (map[agent.id] || map.advisory).map(([tool_id, name_cn]) => ({
    tool_id,
    name_cn,
    success: true,
    elapsed_ms: 42,
  }))
}
</script>

<style scoped>
.console-grid {
  max-width: 1520px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(320px, 392px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.command-panel,
.analysis-stage {
  border: 1px solid #d7ded8;
  background: #fbfcfa;
}

.command-panel {
  padding: 16px;
}

.panel-head {
  height: 34px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.panel-head span,
.field-block label {
  color: #314139;
  font-size: 13px;
  font-weight: 700;
}

.quick-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 10px 0 16px;
}

.quick-row button {
  height: 34px;
  border: 1px solid #ccd7d0;
  background: #f3f6f2;
  color: #304039;
  cursor: pointer;
}

.quick-row button:hover {
  border-color: #8d7650;
  color: #6d572f;
}

.field-block {
  display: grid;
  gap: 8px;
  margin-top: 14px;
}

.profile-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.submit-row {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
}

.notice {
  margin-top: 12px;
}

.router-banner {
  margin: 0 0 14px;
  max-width: 1520px;
}

.analysis-stage {
  min-width: 0;
}

.process-strip {
  min-height: 58px;
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  border-bottom: 1px solid #d7ded8;
  background: #f7f8f5;
}

.process-step {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0 14px;
  border-right: 1px solid #d7ded8;
}

.process-step:last-child {
  border-right: 0;
}

.process-step span {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #b8c3bb;
}

.process-step.done span {
  background: #287653;
}

.process-step.active span {
  background: #c49b54;
  box-shadow: 0 0 0 4px rgba(196, 155, 84, 0.14);
}

.process-step p {
  margin: 0;
  color: #48584f;
  font-size: 13px;
  font-weight: 650;
}

@media (max-width: 1180px) {
  .console-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 680px) {
  .profile-grid,
  .process-strip {
    grid-template-columns: 1fr;
  }

  .process-step {
    min-height: 44px;
    border-right: 0;
    border-bottom: 1px solid #d7ded8;
  }
}
</style>
