import FaceBoxes from './FaceBoxes.jsx'
import ImageCard from './ImageCard.jsx'

// Turn "[1]" / "[2][3]" citations in the answer into buttons that open the photo.
function AnswerText({ text, onCite, streaming }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <p className={`answer ${streaming ? 'streaming' : ''}`}>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/)
        if (!m) return part
        const n = Number(m[1])
        return (
          <button key={i} type="button" className="cite" onClick={() => onCite(n - 1)}>
            {n}
          </button>
        )
      })}
      {streaming && <span className="caret" aria-hidden="true" />}
    </p>
  )
}

function PipelineInfo({ msg, onBrowse }) {
  const p = msg.parsed
  const bits = []
  if (p) {
    if (p.names?.length) bits.push(`person: ${p.names.join(' + ')}${p.inherited ? ' (from previous question)' : ''}`)
    else if (p.lookup && p.lookup_is_name) bits.push(`person: “${p.lookup}” (not found)`)
    else bits.push('person: none detected → semantic search')
    if (p.years?.length > 1) {
      bits.push(`years: ${p.years.join(', ')}`)
    } else if (p.year_from || p.year_to) {
      bits.push(
        p.year_from === p.year_to ? `year: ${p.year_from}` : `years: ${p.year_from ?? '…'}–${p.year_to ?? '…'}`,
      )
    }
  }
  bits.push(`${msg.images.length} photo${msg.images.length === 1 ? '' : 's'}`)
  if (msg.streaming) bits.push(`search ${msg.timing?.retrieve ?? 0}s · writing answer…`)
  else if (msg.timing) bits.push(`search ${msg.timing.retrieve}s · answer ${msg.timing.generate}s`)
  if (p?.new_person) bits.push(`new person: ${p.new_person}, face learned from Wikidata + Commons`)
  if (p?.lookup_error && !p.people?.length) bits.push(`lookup “${p.lookup}”: ${p.lookup_error}`)
  if (msg.live?.added) bits.push(`+${msg.live.added} new from Wikimedia Commons`)
  else if (msg.live?.reason?.includes('offline')) bits.push('Wikimedia Commons unreachable: archive only')
  if (msg.llm === false) bits.push('LLM offline – simple summary')
  const one = p?.people?.length === 1 ? p.people[0] : null
  return (
    <div className="pipeline">
      {bits.join(' · ')}
      {one && !msg.streaming && onBrowse && (
        <button type="button" className="see-all" onClick={() => onBrowse(one)}>
          See all photos of {p.names[0]} →
        </button>
      )}
    </div>
  )
}

// While someone is learned in the background, App.jsx polls the server; this says what's going on.
function LearningNote({ msg }) {
  const p = msg.parsed
  if (!p?.lookup_pending) return null
  const st = msg.learning
  if (st?.state === 'unknown') {
    return <div className="learning stopped">Background learning stopped (the server restarted). Ask again to start it.</div>
  }
  if (msg.learnDone) return null
  const who = st?.label || p.lookup_pending
  const confirmed = st?.confirmed ?? p.lookup_confirmed
  return (
    <div className="learning" role="status">
      <span className="pulse" aria-hidden="true" />
      {confirmed ? `Learning ${who}’s face` : `Looking up “${who}”`} in the background
      {st?.minutes ? ` · ${st.minutes} min so far` : ''} · photos will appear here when it’s done
    </div>
  )
}

export default function Message({ msg, onOpen, onBrowse }) {
  if (msg.role === 'user') {
    return (
      <div className="msg user">
        {msg.upload ? (
          <div className="bubble upload">
            <FaceBoxes src={msg.upload} faces={msg.faces || []} className="small" />
            <span>{msg.text}</span>
          </div>
        ) : (
          <div className="bubble">{msg.text}</div>
        )}
      </div>
    )
  }

  if (msg.role === 'error') {
    return (
      <div className="msg assistant">
        <div className="error">{msg.text}</div>
      </div>
    )
  }

  if (msg.pending) {
    return (
      <div className="msg assistant">
        <div className="thinking">
          <span />
          <span />
          <span />
          {msg.status || msg.text}
        </div>
      </div>
    )
  }

  return (
    <div className="msg assistant">
      {msg.streaming && !msg.text ? (
        <div className="thinking">
          <span />
          <span />
          <span />
          {msg.status || 'Writing answer…'}
        </div>
      ) : (
        <AnswerText
          text={msg.text}
          streaming={msg.streaming}
          onCite={(i) => msg.images[i] && onOpen(msg.images, i)}
        />
      )}
      <LearningNote msg={msg} />
      <PipelineInfo msg={msg} onBrowse={onBrowse} />
      {msg.images.length > 0 && (
        <div className="grid">
          {msg.images.map((img, i) => (
            <ImageCard key={img.id} image={img} index={i + 1} onOpen={() => onOpen(msg.images, i)} />
          ))}
        </div>
      )}
    </div>
  )
}
