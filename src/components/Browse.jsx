import { useEffect, useState } from 'react'
import { api } from '../api.js'
import FaceThumb from './FaceThumb.jsx'
import ImageCard from './ImageCard.jsx'

const PAGE = 60

// "All photos": every indexed photo, filtered by person and year.
export default function Browse({ person, onPersonChange, people, onOpen }) {
  const [year, setYear] = useState('')
  const [sort, setSort] = useState('newest')
  const key = `${person}|${year}|${sort}`
  const [result, setResult] = useState({ key: null, data: null, items: [], error: null })
  const [moreLoading, setMoreLoading] = useState(false)

  // new filters -> first page (the old results stay on screen, dimmed, until it arrives)
  useEffect(() => {
    let cancelled = false
    api
      .images({ person, year, sort, offset: 0, limit: PAGE })
      .then((res) => !cancelled && setResult({ key, data: res, items: res.items, error: null }))
      .catch((e) => !cancelled && setResult({ key, data: null, items: [], error: e.message }))
    return () => {
      cancelled = true
    }
  }, [key, person, year, sort])

  const { data, items, error } = result
  const stale = result.key !== key
  const loading = stale || moreLoading

  const loadMore = async () => {
    setMoreLoading(true)
    try {
      const res = await api.images({ person, year, sort, offset: items.length, limit: PAGE })
      setResult((cur) => ({ ...cur, items: [...cur.items, ...res.items] }))
    } catch (e) {
      setResult((cur) => ({ ...cur, error: e.message }))
    } finally {
      setMoreLoading(false)
    }
  }

  const thumbs = Object.fromEntries(people.map((p) => [p.id, p]))
  const chips = data?.people ?? []
  const selectedName = chips.find((c) => c.id === person)?.name

  return (
    <main className="browse">
      <div className="browse-head">
        <h2>
          {selectedName ? `Photos of ${selectedName}` : 'All photos'}
          {data && <span className="count">{data.total}</span>}
        </h2>
        <div className="browse-controls">
          <select value={year} onChange={(e) => setYear(e.target.value)} aria-label="Year">
            <option value="">All years</option>
            {data?.years.map((y) => (
              <option key={y.year} value={y.year}>
                {y.year === 'none' ? 'No date' : y.year} ({y.count})
              </option>
            ))}
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort">
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="added">Recently added</option>
          </select>
        </div>
      </div>

      <div className="browse-people">
        <button type="button" className={`person ${!person ? 'active' : ''}`} onClick={() => onPersonChange('')}>
          <span>Everyone</span>
          {data && <small>{data.all_count}</small>}
        </button>
        {chips
          .filter((c) => c.count > 0 || c.id === person)
          .map((c) => (
            <button
              key={c.id}
              type="button"
              className={`person ${person === c.id ? 'active' : ''}`}
              onClick={() => onPersonChange(person === c.id ? '' : c.id)}
            >
              {thumbs[c.id] && (
                <FaceThumb
                  src={thumbs[c.id].thumb}
                  box={thumbs[c.id].thumb_box}
                  imageSize={thumbs[c.id].thumb_size}
                  size={28}
                  alt={c.name}
                />
              )}
              <span>{c.name}</span>
              <small>{c.count}</small>
            </button>
          ))}
      </div>

      {error && <div className="error">{error}</div>}

      {data && items.length === 0 && !stale && (
        <p className="empty-note">
          No photos match these filters. {year && 'Try "All years".'}
        </p>
      )}

      <div className={`grid ${stale && items.length ? 'stale' : ''}`}>
        {items.map((img, i) => (
          <ImageCard key={img.id} image={img} onOpen={() => onOpen(items, i)} />
        ))}
      </div>

      {data && items.length < data.total && (
        <div className="load-more">
          <button type="button" className="counter" onClick={loadMore} disabled={loading}>
            {moreLoading ? 'Loading…' : `Show more (${data.total - items.length} left)`}
          </button>
        </div>
      )}
      {stale && !items.length && !error && <div className="thinking browse-loading"><span /><span /><span />Loading photos…</div>}
    </main>
  )
}
