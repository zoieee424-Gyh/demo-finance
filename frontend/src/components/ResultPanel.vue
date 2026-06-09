<template>
  <section class="result-layout">
    <article class="report-paper">
      <div v-if="loading" class="loading-block">
        <el-skeleton :rows="12" animated />
      </div>

      <template v-else>
        <header class="report-head">
          <div>
            <span class="report-kicker">{{ intentLabel }}</span>
            <h2>{{ result?.title || '分析报告' }}</h2>
          </div>
          <div class="report-stamp">
            <span>{{ architectureLabel }}</span>
            <strong>{{ debug?.fallback_used ? 'Fallback' : 'Primary' }}</strong>
          </div>
        </header>

        <div class="report-body">
          <section
            v-for="section in reportSections"
            :key="section.title"
            class="report-section"
          >
            <h3>{{ section.title }}</h3>
            <p>{{ section.content }}</p>
          </section>
        </div>

        <footer v-if="result?.risk_notice" class="risk-note">
          <strong>风险提示</strong>
          <span>{{ result.risk_notice }}</span>
        </footer>
      </template>
    </article>

    <aside class="detail-pane">
      <section class="metric-band">
        <div>
          <span>{{ sourceCount }}</span>
          <label>Sources</label>
        </div>
        <div>
          <span>{{ debug?.tool_count || traceCount }}</span>
          <label>Tools</label>
        </div>
      </section>

      <section class="side-section">
        <h3>工具轨迹</h3>
        <ul class="trace-list">
          <li v-for="trace in traces" :key="trace.tool_id">
            <span class="trace-dot" :class="{ failed: trace.success === false }"></span>
            <div>
              <strong>{{ trace.name_cn || trace.tool_id }}</strong>
              <small>{{ trace.tool_id }}</small>
            </div>
          </li>
        </ul>
      </section>

      <section class="side-section">
        <h3>证据来源</h3>
        <div class="source-list">
          <div v-for="source in formattedSources" :key="source.title" class="source-item">
            <strong>{{ source.title }}</strong>
            <span>{{ source.source_type }} · {{ source.confidenceLabel }}</span>
          </div>
        </div>
      </section>

      <section class="side-section debug-section">
        <h3>运行信息</h3>
        <dl>
          <dt>Endpoint</dt>
          <dd>{{ debug?.endpoint || '-' }}</dd>
          <dt>Architecture</dt>
          <dd>{{ debug?.actual_architecture || 'deepagent' }}</dd>
          <dt>Fallback</dt>
          <dd>{{ debug?.fallback_used ? 'true' : 'false' }}</dd>
        </dl>
      </section>
    </aside>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  agent: {
    type: Object,
    required: true,
  },
  result: {
    type: Object,
    default: null,
  },
  debug: {
    type: Object,
    default: null,
  },
  loading: {
    type: Boolean,
    default: false,
  },
})

const sourceCount = computed(() => props.result?.sources?.length || 0)
const traces = computed(() => props.debug?.tool_traces || [])
const traceCount = computed(() => traces.value.length)
const architectureLabel = computed(() => props.debug?.actual_architecture || 'deepagent')
const intentLabel = computed(() => props.agent?.name || props.result?.intent || '智能体')

const reportSections = computed(() => {
  const answer = props.result?.answer || ''
  if (!answer.trim()) return []

  const lines = answer.split(/\n+/).map((line) => line.trim()).filter(Boolean)
  const sections = []
  let current = null

  for (const line of lines) {
    const normalized = line.replace(/^#{1,4}\s*/, '').replace(/^\*\*|\*\*$/g, '')
    if (/^[一二三四五六七八九十]、/.test(normalized)) {
      current = { title: normalized, content: '' }
      sections.push(current)
    } else if (current) {
      current.content += `${current.content ? '\n' : ''}${normalized}`
    } else {
      current = { title: '摘要', content: normalized }
      sections.push(current)
    }
  }

  return sections.map((section) => ({
    ...section,
    content: section.content || '当前材料未提供充分信息。',
  }))
})

const formattedSources = computed(() => (props.result?.sources || []).map((item) => ({
  ...item,
  confidenceLabel: typeof item.confidence === 'number'
    ? `${Math.round(item.confidence * 100)}%`
    : 'N/A',
})))
</script>

<style scoped>
.result-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 286px;
  min-height: 660px;
}

.report-paper {
  min-width: 0;
  padding: 24px 28px 26px;
  background: #fbfcfa;
  border-right: 1px solid #d7ded8;
}

.loading-block {
  padding: 18px 0;
}

.report-head {
  min-height: 72px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding-bottom: 18px;
  border-bottom: 1px solid #d7ded8;
}

.report-kicker {
  color: #7b6847;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

h2 {
  margin: 6px 0 0;
  color: #16231d;
  font-size: 24px;
  line-height: 1.25;
  letter-spacing: 0;
}

.report-stamp {
  width: 108px;
  height: 54px;
  display: grid;
  place-items: center;
  align-content: center;
  border: 1px solid #b99d68;
  color: #6b562f;
  background: #faf7ef;
}

.report-stamp span,
.report-stamp strong {
  display: block;
}

.report-stamp span {
  font-size: 11px;
}

.report-stamp strong {
  margin-top: 3px;
  font-size: 13px;
}

.report-body {
  display: grid;
  gap: 18px;
  padding-top: 20px;
}

.report-section {
  padding-left: 14px;
  border-left: 3px solid #9b8a62;
}

.report-section h3 {
  margin: 0 0 8px;
  color: #17211c;
  font-size: 16px;
  letter-spacing: 0;
}

.report-section p {
  margin: 0;
  white-space: pre-wrap;
  color: #34463d;
  font-size: 14px;
  line-height: 1.85;
}

.risk-note {
  display: grid;
  gap: 6px;
  margin-top: 24px;
  padding: 14px 16px;
  background: #f8f3e7;
  border: 1px solid #ded0ad;
  color: #5e4a24;
}

.risk-note strong {
  font-size: 13px;
}

.risk-note span {
  font-size: 13px;
  line-height: 1.6;
}

.detail-pane {
  min-width: 0;
  background: #f3f6f2;
}

.metric-band {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-bottom: 1px solid #d7ded8;
}

.metric-band div {
  height: 72px;
  display: grid;
  place-items: center;
  align-content: center;
  border-right: 1px solid #d7ded8;
}

.metric-band div:last-child {
  border-right: 0;
}

.metric-band span {
  color: #16231d;
  font-size: 22px;
  font-weight: 760;
  line-height: 1;
}

.metric-band label {
  margin-top: 6px;
  color: #6c7d73;
  font-size: 11px;
  text-transform: uppercase;
}

.side-section {
  padding: 16px;
  border-bottom: 1px solid #d7ded8;
}

.side-section h3 {
  margin: 0 0 12px;
  color: #314139;
  font-size: 13px;
  letter-spacing: 0;
}

.trace-list {
  display: grid;
  gap: 12px;
  list-style: none;
  padding: 0;
  margin: 0;
}

.trace-list li {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}

.trace-dot {
  width: 8px;
  height: 8px;
  margin-top: 5px;
  border-radius: 50%;
  background: #287653;
}

.trace-dot.failed {
  background: #b75d4e;
}

.trace-list strong,
.trace-list small {
  display: block;
}

.trace-list strong {
  color: #1e2e27;
  font-size: 13px;
}

.trace-list small {
  margin-top: 3px;
  color: #708176;
  font-size: 11px;
  word-break: break-all;
}

.source-list {
  display: grid;
  gap: 10px;
}

.source-item {
  display: grid;
  gap: 4px;
  padding: 10px;
  background: #fbfcfa;
  border: 1px solid #d7ded8;
}

.source-item strong {
  color: #1e2e27;
  font-size: 13px;
  line-height: 1.45;
}

.source-item span {
  color: #708176;
  font-size: 12px;
}

dl {
  display: grid;
  gap: 8px;
  margin: 0;
}

dt {
  color: #708176;
  font-size: 11px;
}

dd {
  margin: -4px 0 2px;
  color: #1e2e27;
  font-size: 12px;
  word-break: break-all;
}

@media (max-width: 860px) {
  .result-layout {
    grid-template-columns: 1fr;
  }

  .report-paper {
    border-right: 0;
    border-bottom: 1px solid #d7ded8;
  }
}

@media (max-width: 620px) {
  .report-paper {
    padding: 18px;
  }

  .report-head {
    flex-direction: column;
  }
}
</style>
