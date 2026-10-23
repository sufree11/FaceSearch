import { useRef, useState } from 'react'

export default function Composer({ value, onChange, onSubmit, onUpload, busy, big = false, autoFocus }) {
  const fileRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const pickFile = (file) => {
    if (file && file.type.startsWith('image/')) onUpload(file)
  }

  return (
    <form
      className={`composer ${big ? 'big' : ''} ${dragging ? 'dragging' : ''}`}
      onSubmit={(e) => {
        e.preventDefault()
        if (value.trim() && !busy) onSubmit(value.trim())
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        pickFile(e.dataTransfer.files?.[0])
      }}
    >
      <button
        type="button"
        className="icon-btn"
        title="Search by face: upload a photo"
        aria-label="Search by face"
        onClick={() => fileRef.current?.click()}
        disabled={busy}
      >
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2" />
          <circle cx="12" cy="10.5" r="3" />
          <path d="M7.5 17c1-2 2.6-3 4.5-3s3.5 1 4.5 3" />
        </svg>
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          pickFile(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      <input
        type="text"
        id="search"
        autoFocus={autoFocus}
        autoComplete="off"
        placeholder={dragging ? 'Drop a photo to search by face…' : 'Cth: "Bagi gambar Tun Mahathir"'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onPaste={(e) => pickFile([...e.clipboardData.files][0])}
      />
      <button type="submit" className="send" disabled={busy || !value.trim()}>
        {busy ? '…' : 'Search'}
      </button>
    </form>
  )
}
