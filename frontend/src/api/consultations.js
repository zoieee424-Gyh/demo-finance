const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

const DEFAULT_TIMEOUT_MS = 240000  // 240s — DeepAgent can take 30-120s+ per request

/**
 * POST JSON with timeout and abort support.
 *
 * @param {string} path — API path (e.g. /api/debug/investment-advisor)
 * @param {object} payload — JSON body
 * @param {object} [options]
 * @param {AbortSignal} [options.signal] — external AbortSignal
 * @param {number} [options.timeoutMs] — timeout in ms (default 240000)
 * @returns {Promise<object>} parsed JSON response
 */
async function postJson(path, payload, options = {}) {
  const { signal: externalSignal, timeoutMs = DEFAULT_TIMEOUT_MS } = options

  const controller = new AbortController()
  const signal = controller.signal

  // Wire external signal
  if (externalSignal) {
    if (externalSignal.aborted) {
      const err = new DOMException('Request cancelled', 'AbortError')
      err.name = 'AbortError'
      throw err
    }
    externalSignal.addEventListener('abort', () => controller.abort())
  }

  const timer = setTimeout(() => {
    controller.abort()
  }, timeoutMs)

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    })

    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || `Request failed: ${response.status}`)
    }

    return response.json()
  } catch (err) {
    if (err.name === 'AbortError') {
      if (externalSignal?.aborted) {
        // Externally cancelled (user switched agent, etc.)
        throw err
      }
      // Timeout — internal abort
      const timeoutErr = new Error(
        '请求超时。DeepAgent 可能仍在多轮调用工具处理中，请稍后重试或简化输入内容。'
      )
      timeoutErr.name = 'TimeoutError'
      throw timeoutErr
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export function createConsultation(payload) {
  return postJson('/api/consultations', payload)
}

export function debugInvestmentAdvisor(payload) {
  return postJson('/api/debug/investment-advisor', payload)
}

/**
 * Unified debug agent caller — used by all five agent buttons.
 *
 * @param {string} endpoint — e.g. /api/debug/financial-report
 * @param {object} payload — { question, user_profile }
 * @param {object} [options] — { signal, timeoutMs }
 */
export function postDebugAgent(endpoint, payload, options) {
  return postJson(endpoint, payload, options)
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`)
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`)
  }
  return response.json()
}

// ── SSE Streaming ──────────────────────────────────────────────

/**
 * Prepare a streaming run.
 *
 * @param {string} agentId — "advisory" | "financial_report" | "risk_control" | "compliance" | "education"
 * @param {object} payload — { question, user_profile }
 * @returns {Promise<{run_id: string, stream_url: string, agent_id: string}>}
 */
export async function prepareStreamAgent(agentId, payload) {
  return postJson(`/api/debug/stream/${agentId}/prepare`, payload)
}

/**
 * Open an EventSource to a stream URL.
 *
 * Returns a controller object with:
 *   - events: the EventSource (addEventListener to it)
 *   - close(): close the connection
 *
 * @param {string} streamUrl — e.g. "/api/debug/stream/{run_id}"
 * @returns {{ events: EventSource, close: () => void }}
 */
export function openAgentStream(streamUrl) {
  const es = new EventSource(`${API_BASE}${streamUrl}`)
  return {
    events: es,
    close: () => es.close(),
  }
}
