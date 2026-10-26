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

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealthError(true))
    api.people().then(setPeople).catch(() => {})
  }, [])

  // scroll only when a message is added, not on every streamed word,
  // so the reply being written stays in view
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length])

  const replacePending = (msg) => setMessages((m) => [...m.slice(0, -1), msg])
  const updateLast = (fn) => setMessages((m) => [...m.slice(0, -1), fn(m[m.length - 1])])

  const ask = useCallback(
    async (text) => {
      setQuery('')
      setBusy(true)
      const history = messages
        .filter((m) => m.role === 'assistant' && !m.pending && !m.streaming)
        .map((m) => ({ role: 'assistant', content: m.text, parsed: m.parsed }))
      setMessages((m) => [
        ...m,
        { role: 'user', text },
        { role: 'assistant', pending: true, text: 'Searching faces & descriptions…' },
      ])
      try {
        await api.chatStream(text, history, (ev) => {
          if (ev.type === 'results') {
            // photos appear straight away; the reply is written underneath.
            // A second "results" comes after a live fetch from Wikimedia Commons.
            const update = { images: ev.images, parsed: ev.parsed, timing: ev.timing, live: ev.live, status: null }
            if (ev.parsed?.new_person || ev.live?.added) {
              // the archive just grew: refresh the people list and the status counts
              api.people().then(setPeople).catch(() => {})
              api.health().then(setHealth).catch(() => {})
            }
            setMessages((all) => {
              const last = all[all.length - 1]
              const next = last?.streaming
                ? { ...last, ...update }
                : { role: 'assistant', text: '', streaming: true, ...update }
              return [...all.slice(0, -1), next]
            })
          } else if (ev.type === 'status') {
            updateLast((m) => ({ ...m, status: ev.text }))
          } else if (ev.type === 'token') {
            updateLast((m) => ({ ...m, text: m.text + ev.text }))
          } else if (ev.type === 'done') {
            updateLast((m) => ({ ...m, text: ev.answer, timing: ev.timing, llm: ev.llm, streaming: false }))
          } else if (ev.type === 'error') {
            replacePending({ role: 'error', text: ev.message })
          }
        })
        // stream ended without a "done" event (e.g. backend stopped): stop the cursor
        updateLast((m) => (m.streaming ? { ...m, streaming: false } : m))
      } catch (e) {
        replacePending({ role: 'error', text: e.message })
      } finally {
        setBusy(false)
      }
    },
    [messages],
  )

  const searchByFace = async (file) => {
    setBusy(true)
    const preview = URL.createObjectURL(file)
    setMessages((m) => [
      ...m,
      { role: 'user', text: 'Who is this? Find more photos.', upload: preview },
      { role: 'assistant', pending: true, text: 'Detecting and matching faces…' },
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
      replacePending({ role: 'assistant', text, images: res.images, parsed: null })
    } catch (e) {
      replacePending({ role: 'error', text: e.message })
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
          {messages.map((msg, i) => (
            <Message key={i} msg={msg} onOpen={openViewer} onBrowse={openBrowse} />
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
