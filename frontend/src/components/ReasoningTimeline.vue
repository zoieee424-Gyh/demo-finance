<template>
  <section class="timeline-section" v-if="events.length > 0">
    <h3>推理链</h3>
    <div class="timeline-list">
      <div
        v-for="(event, idx) in events"
        :key="idx"
        class="timeline-item"
        :class="[event._state || 'done']"
      >
        <span class="timeline-dot" :class="event._state || 'done'"></span>
        <div class="timeline-body">
          <div class="timeline-head">
            <strong>{{ event.label }}</strong>
            <small v-if="event._time">{{ event._time }}</small>
          </div>
          <p v-if="event.message && event._state !== 'done'">{{ event.message }}</p>
          <p v-if="toolLabel(event)" class="timeline-tool">
            <code>{{ event.tool_id }}</code>
            <span>{{ toolLabel(event) }}</span>
            <span :class="toolSuccess(event) ? 'ok' : 'fail'">
              {{ toolSuccess(event) ? '✓' : '✗' }}
            </span>
            <small v-if="event.elapsed_ms">{{ event.elapsed_ms.toFixed(0) }}ms</small>
          </p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const props = defineProps({
  events: {
    type: Array,
    default: () => [],
  },
})

// 同时兼容新实时事件(name_cn/success)和旧补发事件(tool_name/tool_success)。
function toolLabel(event) {
  return event.name_cn || event.tool_name || ''
}

function toolSuccess(event) {
  return event.success ?? event.tool_success ?? true
}
</script>

<style scoped>
.timeline-section {
  padding: 16px;
  border-bottom: 1px solid #d7ded8;
}

.timeline-section h3 {
  margin: 0 0 12px;
  color: #314139;
  font-size: 13px;
  letter-spacing: 0;
}

.timeline-list {
  display: grid;
  gap: 0;
}

.timeline-item {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr);
  gap: 8px;
  padding: 6px 0 6px 2px;
  border-left: 2px solid #d7ded8;
}

.timeline-item:first-child {
  border-left-color: transparent;
}

.timeline-item.active {
  border-left-color: #c49b54;
}

.timeline-dot {
  width: 9px;
  height: 9px;
  margin-top: 4px;
  margin-left: -7px;
  border-radius: 50%;
  background: #287653;
  border: 2px solid #fbfcfa;
  z-index: 1;
}

.timeline-dot.active {
  background: #c49b54;
  box-shadow: 0 0 0 3px rgba(196, 155, 84, 0.2);
}

.timeline-dot.error {
  background: #b75d4e;
}

.timeline-body strong {
  display: block;
  color: #1e2e27;
  font-size: 13px;
  font-weight: 650;
}

.timeline-body p {
  margin: 4px 0 0;
  color: #48584f;
  font-size: 12px;
  line-height: 1.55;
}

.timeline-body small {
  color: #708176;
  font-size: 11px;
}

.timeline-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
}

.timeline-tool {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
}

.timeline-tool code {
  font-size: 11px;
  background: #f3f6f2;
  padding: 1px 5px;
  border-radius: 2px;
  color: #48584f;
}

.timeline-tool .ok {
  color: #287653;
  font-weight: 700;
}

.timeline-tool .fail {
  color: #b75d4e;
  font-weight: 700;
}
</style>
