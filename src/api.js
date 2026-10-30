// All calls go to the local Python backend (proxied by Vite, see vite.config.js).

async function request(path, options = {}) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = body.detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  people: () => request('/api/people'),
  random: () => request('/api/random'),
  // is a person still being learned in the background? {state: 'pending'|'done'|'failed'|'unknown', label, minutes}
  learning: (key) => request(`/api/learning?${new URLSearchParams({ key })}`),
  images: (params) => request(`/api/images?${new URLSearchParams(params)}`),
  // Streams the reply: calls onEvent({type: 'results' | 'token' | 'done' | 'error', ...}) as events arrive.
  chatStream: async (message, history, onEvent) => {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
    })
    if (!res.ok || !res.body) {
      let detail = `${res.status} ${res.statusText}`
      try {
        detail = (await res.json()).detail || detail
      } catch {
        /* not JSON */
      }
      throw new Error(detail)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() // keep a half-received line for the next chunk
      for (const line of lines) if (line.trim()) onEvent(JSON.parse(line))
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer))
  },
  faceSearch: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/face-search', { method: 'POST', body: form })
  },
}
