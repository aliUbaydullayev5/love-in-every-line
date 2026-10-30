// Падает, если в dist/ попали фото или промежуточные файлы (приватность).
import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const bad = []
const walk = (d) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f)
    if (statSync(p).isDirectory()) walk(p)
    else if (/\.(jpe?g|png|webp|bmp|heic|tmp)$/i.test(f) || /\.src\./.test(f)) bad.push(p)
  }
}
walk('dist')
if (bad.length) {
  console.error('В dist/ попали файлы, которых там быть не должно:\n' + bad.join('\n'))
  process.exit(1)
}
console.log('dist/: фотографий и временных файлов нет')
