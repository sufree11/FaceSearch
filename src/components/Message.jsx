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
    bits.push(p.names?.length ? `person: ${p.names.join(' + ')}${p.inherited ? ' (from previous question)' : ''}` : 'person: none detected → semantic search')
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
          {msg.text}
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
          Writing answer…
        </div>
      ) : (
        <AnswerText
          text={msg.text}
          streaming={msg.streaming}
          onCite={(i) => msg.images[i] && onOpen(msg.images, i)}
        />
      )}
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
