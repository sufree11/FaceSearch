import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import Browse from './components/Browse.jsx'
import Composer from './components/Composer.jsx'
import FaceThumb from './components/FaceThumb.jsx'
import Lightbox from './components/Lightbox.jsx'
import Message from './components/Message.jsx'
import './App.css'

function StatusPill({ health, error }) {
  if (error) return <span className="status bad">Backend offline: start the Python server</span>
  if (!health) return <span className="status">Checking…</span>
  const { ollama, index } = health
  if (!index.images) return <span className="status warn">No photos indexed yet</span>
  const missing = Object.values(ollama.models).filter((m) => !m.ready)
  if (!ollama.running) return <span className="status warn">Ollama offline: face search only</span>
  if (missing.length) return <span className="status warn">Missing model: {missing.map((m) => m.name).join(', ')}</span>
  return (
    <span className="status ok">
      Local AI online
    </span>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)
  const [people, setPeople] = useState([])
  const [viewer, setViewer] = useState(null) // { images, index }
  const [view, setView] = useState('chat') // 'chat' | 'browse'
  const [browsePerson, setBrowsePerson] = useState('')
  const endRef = useRef(null)
  const nextId = useRef(1)
  const newId = () => nextId.current++

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealthError(true))
    api.people().then(setPeople).catch(() => {})
  }, [])

  // scroll only when a message is added, not on every streamed word,
  // so the reply being written stays in view
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length])

  // every message has an id, so an answer can be filled in later (see the learning watcher below)
  // even after more questions were asked
  const updateById = (id, fn) => setMessages((all) => all.map((m) => (m.id === id ? fn(m) : m)))
  const replaceLast = (msg) => setMessages((m) => [...m.slice(0, -1), { id: newId(), ...msg }])

  // ask(text): a new question. ask(text, { retryId }): run it again into an existing answer,
  // used when a person finished being learned in the background.
  const ask = useCallback(
    async (text, { retryId, readyLabel } = {}) => {
      const id = retryId ?? newId()
      if (!retryId) {
        setQuery('')
        setBusy(true)
      }
      const history = messages
        .filter((m) => m.role === 'assistant' && !m.pending && !m.streaming && m.id !== id)
        .map((m) => ({ role: 'assistant', content: m.text, parsed: m.parsed }))
      const waiting = {
        id,
        role: 'assistant',
        pending: true,
        question: text,
        text: readyLabel ? `${readyLabel} is ready, searching again…` : 'Searching faces & descriptions…',
      }
      if (retryId) updateById(id, () => waiting)
      else setMessages((m) => [...m, { id: newId(), role: 'user', text }, waiting])
      const fail = (message) => updateById(id, () => ({ id, role: 'error', text: message }))
      try {
        await api.chatStream(text, history, (ev) => {
          if (ev.type === 'results') {
            // photos appear straight away; the reply is written underneath.
            // A second "results" comes after a live fetch from Wikimedia Commons.
            const update = { images: ev.images, parsed: ev.parsed, timing: ev.timing, live: ev.live, status: null }
            if (ev.parsed?.new_person || ev.live?.added || retryId) {
              // the archive just grew: refresh the people list and the status counts
              api.people().then(setPeople).catch(() => {})
              api.health().then(setHealth).catch(() => {})
            }
            updateById(id, (m) =>
              m.streaming
                ? { ...m, ...update }
                : { id, role: 'assistant', question: text, text: '', streaming: true, ...update },
            )
          } else if (ev.type === 'status') {
            updateById(id, (m) => ({ ...m, status: ev.text }))
          } else if (ev.type === 'token') {
            updateById(id, (m) => ({ ...m, text: m.text + ev.text }))
          } else if (ev.type === 'done') {
            updateById(id, (m) => ({ ...m, text: ev.answer, timing: ev.timing, llm: ev.llm, streaming: false }))
          } else if (ev.type === 'error') {
            fail(ev.message)
          }
        })
        // stream ended without a "done" event (e.g. backend stopped): stop the cursor
        updateById(id, (m) => (m.streaming ? { ...m, streaming: false } : m))
      } catch (e) {
        fail(e.message)
      } finally {
        if (!retryId) setBusy(false)
      }
    },
    [messages],
  )

  // Someone is being learned in the background ("I'm learning X's face…"): ask the server every
  // few seconds, and when it's finished run the question again into the same answer, so the
  // photos appear without asking twice.
  const messagesRef = useRef(messages)
  const askRef = useRef(ask)
  useEffect(() => {
    messagesRef.current = messages
    askRef.current = ask
  }, [messages, ask])
  const watched = messages.filter((m) => m.role === 'assistant' && !m.pending && !m.streaming && m.parsed?.lookup_pending && m.parsed?.lookup_key && !m.learnDone)
  const watchedIds = watched.map((m) => m.id).join(',')
  useEffect(() => {
    if (!watchedIds) return undefined
    const tick = async () => {
      const current = messagesRef.current.filter((m) => watchedIds.split(',').includes(String(m.id)))
      for (const m of current) {
        if (m.learnDone || m.pending || m.streaming) continue
        let st
        try {
          st = await api.learning(m.parsed.lookup_key)
        } catch {
          continue // backend briefly unreachable: try again next tick
        }
        if (st.state === 'pending') {
          updateById(m.id, (x) => ({ ...x, learning: st }))
        } else if (st.state === 'unknown') {
          updateById(m.id, (x) => ({ ...x, learnDone: true, learning: st })) // e.g. server restarted
        } else {
          const label = m.learning?.label || m.parsed.lookup_pending
          updateById(m.id, (x) => ({ ...x, learnDone: true }))
          askRef.current(m.question, { retryId: m.id, readyLabel: st.state === 'done' ? label : null })
        }
      }
    }
    const timer = setInterval(tick, 5000)
    return () => clearInterval(timer)
  }, [watchedIds])

  const searchByFace = async (file) => {
    setBusy(true)
    const preview = URL.createObjectURL(file)
    setMessages((m) => [
      ...m,
      { id: newId(), role: 'user', text: 'Who is this? Find more photos.', upload: preview },
      { id: newId(), role: 'assistant', pending: true, text: 'Detecting and matching faces…' },
    ])
    try {
      const res = await api.faceSearch(file)
      // show the detected boxes on the uploaded photo
      setMessages((m) => {
        const copy = [...m]
        copy[copy.length - 2] = { ...copy[copy.length - 2], faces: res.faces }
        return copy
      })
      let text
      if (!res.faces.length) text = 'No face found in that photo. Try a clearer, front-facing picture.'
      else {
        const main = res.faces[0]
        text = main.name
          ? `This looks like ${main.name} (similarity ${main.score.toFixed(2)}).`
          : 'This face does not match anyone in the reference gallery.'
        if (res.faces.length > 1) text += ` ${res.faces.length} faces detected; searching with the largest one.`
        text += res.images.length
          ? ` Here are ${res.images.length} archive photos with the same face.`
          : ' No archive photos contain this face.'
      }
      replaceLast({ role: 'assistant', text, images: res.images, parsed: null })
    } catch (e) {
      replaceLast({ role: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const openBrowse = (person = '') => {
    setBrowsePerson(person)
    setView('browse')
    window.scrollTo({ top: 0 })
  }

  const lucky = async () => {
    try {
      const { query: q } = await api.random()
      ask(q)
    } catch {
      setHealthError(true)
    }
  }

  const empty = messages.length === 0
  const openViewer = (images, index) => setViewer({ images, index })

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <button
            type="button"
            className="brand"
            onClick={() => {
              setMessages([])
              setView('chat')
            }}
            title="New search"
          >
            FaceSearch
          </button>
          <nav className="tabs">
            <button type="button" className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>
              Search
            </button>
            <button type="button" className={view === 'browse' ? 'active' : ''} onClick={() => openBrowse(browsePerson)}>
              All photos
            </button>
          </nav>
        </div>
        <StatusPill health={health} error={healthError} />
      </header>
      <div className="ticks"></div>

      {view === 'browse' ? (
        <Browse person={browsePerson} onPersonChange={setBrowsePerson} people={people} onOpen={openViewer} />
      ) : empty ? (
        <section id="center">
          <div className="hero">
            <h1>FaceSearch</h1>
            <p className="tagline">
              Ask for a photo of a public figure. FaceSearch recognises the faces, describes each photo and shows its
              date, all on this computer.
            </p>
          </div>
          <div className="search">
            <label htmlFor="search">Search for a face:</label>
            <Composer value={query} onChange={setQuery} onSubmit={ask} onUpload={searchByFace} busy={busy} big autoFocus />
            <p className="hint">Or drop / paste a photo to find the same face.</p>
          </div>
          {people.length > 0 && (
            <div className="people">
              {people.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="person"
                  onClick={() => ask(`Bagi gambar ${p.aliases?.[0] || p.name}`)}
                  title={`${p.image_count} photos`}
                >
                  <FaceThumb src={p.thumb} box={p.thumb_box} imageSize={p.thumb_size} size={32} alt={p.name} />
                  <span>{p.name}</span>
                  <small>{p.image_count}</small>
                </button>
              ))}
            </div>
          )}
          <div className="landing-actions">
            <button type="button" className="counter" onClick={lucky} disabled={busy}>
              I'm feeling lucky
            </button>
            <button type="button" className="counter ghost" onClick={() => openBrowse()}>
              Browse all photos →
            </button>
          </div>
        </section>
      ) : (
        <main className="chat">
          {messages.map((msg) => (
            <Message key={msg.id} msg={msg} onOpen={openViewer} onBrowse={openBrowse} />
          ))}
          <div ref={endRef} />
        </main>
      )}

      {view === 'chat' && !empty && (
        <div className="dock">
          <Composer value={query} onChange={setQuery} onSubmit={ask} onUpload={searchByFace} busy={busy} autoFocus />
        </div>
      )}

      {viewer && (
        <Lightbox
          images={viewer.images}
          index={viewer.index}
          onClose={() => setViewer(null)}
          onMove={(index) => setViewer((v) => ({ ...v, index }))}
        />
      )}
    </>
  )
}

export default App
