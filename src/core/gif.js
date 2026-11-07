// Animated GIF encoding (GIF89a), for the animated export of the fit's construction (src/ui/story.js). Frames are
// opaque RGBA pixels; they share one global palette of at most 256 colors, exact when the frames hold that few, else
// by median cut. A chart is flat colors plus the blends of antialiasing, so no dithering: it would speckle the flat
// areas to gain little at the edges.
//
// Format: the GIF89a specification (CompuServe, 1990); LZW as there, with variable code sizes up to 12 bits and a
// clear code when the table is full. The looping is Netscape's application extension.

const MAX_CODES = 4096; // GIF's LZW table size (12-bit codes)
const HIST_BITS = 5; // bits per channel of the median cut's histogram
const HIST_SIZE = 1 << (3 * HIST_BITS);
const HIST_SHIFT = 8 - HIST_BITS;
const DISPOSE_NONE = 1; // graphic control extension: leave the frame in place (each frame is whole)

/**
 * A palette for the frames: every color they hold when there are at most maxColors, else maxColors by median cut over
 * a histogram at 5 bits per channel, each color the pixel-weighted mean of its cell. Alpha is ignored (frames are
 * opaque). O(pixels + cells·maxColors).
 *
 * @param {Uint8ClampedArray[]|Uint8Array[]} frames  RGBA pixels, 4 bytes each
 * @param {number} [maxColors]  2 to 256
 * @returns {Uint8Array}  RGB triples, at most maxColors of them
 */
export function gifPalette(frames, maxColors = 256) {
    if (!(maxColors >= 2 && maxColors <= 256))
        throw new RangeError('gifPalette: maxColors from 2 to 256 required');
    const exact = new Set();
    let tooMany = false;
    for (const rgba of frames) {
        for (let p = 0; p < rgba.length && !tooMany; p += 4) {
            exact.add((rgba[p] << 16) | (rgba[p + 1] << 8) | rgba[p + 2]);
            if (exact.size > maxColors)
                tooMany = true;
        }
    }
    if (!tooMany) {
        const out = new Uint8Array(3 * Math.max(exact.size, 1));
        let k = 0;
        for (const c of exact) {
            out[k++] = c >> 16;
            out[k++] = (c >> 8) & 255;
            out[k++] = c & 255;
        }
        return out;
    }
    return medianCut(frames, maxColors);
}

function medianCut(frames, maxColors) {
    const count = new Float64Array(HIST_SIZE);
    const sum = new Float64Array(3 * HIST_SIZE);
    for (const rgba of frames) {
        for (let p = 0; p < rgba.length; p += 4) {
            const cell = histCell(rgba[p], rgba[p + 1], rgba[p + 2]);
            count[cell]++;
            sum[3 * cell] += rgba[p];
            sum[3 * cell + 1] += rgba[p + 1];
            sum[3 * cell + 2] += rgba[p + 2];
        }
    }
    const cells = [];
    for (let c = 0; c < HIST_SIZE; c++) {
        if (count[c] > 0)
            cells.push(c);
    }
    const channel = (cell, ch) => (cell >> (HIST_BITS * (2 - ch))) & ((1 << HIST_BITS) - 1);
    const range = box => {
        let best = { ch: 0, size: -1 };
        for (let ch = 0; ch < 3; ch++) {
            let lo = Infinity;
            let hi = -Infinity;
            for (const cell of box) {
                const v = channel(cell, ch);
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
            if (hi - lo > best.size)
                best = { ch, size: hi - lo };
        }
        return best;
    };
    // Split the box spanning the most along a channel, at the pixel-weighted median: a line's few pixels get their
    // own color as surely as the background's many, since a box is chosen by its span, not its pixels.
    const boxes = [cells];
    while (boxes.length < maxColors) {
        let pick = -1;
        let pickRange = null;
        for (let b = 0; b < boxes.length; b++) {
            if (boxes[b].length < 2)
                continue;
            const r = range(boxes[b]);
            if (pick < 0 || r.size > pickRange.size) {
                pick = b;
                pickRange = r;
            }
        }
        if (pick < 0)
            break;
        const box = boxes[pick].sort((a, b) => channel(a, pickRange.ch) - channel(b, pickRange.ch));
        let total = 0;
        for (const cell of box)
            total += count[cell];
        let acc = 0;
        let cut = 1;
        for (; cut < box.length - 1; cut++) {
            acc += count[box[cut - 1]];
            if (acc >= total / 2)
                break;
        }
        boxes.splice(pick, 1, box.slice(0, cut), box.slice(cut));
    }
    const out = new Uint8Array(3 * boxes.length);
    boxes.forEach((box, k) => {
        let n = 0;
        const s = [0, 0, 0];
        for (const cell of box) {
            n += count[cell];
            for (let ch = 0; ch < 3; ch++)
                s[ch] += sum[3 * cell + ch];
        }
        for (let ch = 0; ch < 3; ch++)
            out[3 * k + ch] = Math.round(s[ch] / n);
    });
    return out;
}

function histCell(r, g, b) {
    return ((r >> HIST_SHIFT) << (2 * HIST_BITS)) | ((g >> HIST_SHIFT) << HIST_BITS) | (b >> HIST_SHIFT);
}

/**
 * Each pixel's palette index: its own color's when the palette has it, else the nearest in RGB (squared distance).
 * Colors are looked up once each (cached), so the cost is O(pixels + distinct colors·palette).
 *
 * @param {Uint8ClampedArray|Uint8Array} rgba  RGBA pixels
 * @param {Uint8Array} palette  RGB triples, as gifPalette returns
 * @param {Map<number, number>} [cache]  color → index, to share between frames with the same palette
 * @returns {Uint8Array}  one index per pixel
 */
export function indexPixels(rgba, palette, cache = new Map()) {
    const k = palette.length / 3;
    const out = new Uint8Array(rgba.length / 4);
    for (let p = 0, i = 0; p < rgba.length; p += 4, i++) {
        const r = rgba[p];
        const g = rgba[p + 1];
        const b = rgba[p + 2];
        const key = (r << 16) | (g << 8) | b;
        let idx = cache.get(key);
        if (idx === undefined) {
            let best = Infinity;
            idx = 0;
            for (let j = 0; j < k; j++) {
                const d = (palette[3 * j] - r) ** 2 + (palette[3 * j + 1] - g) ** 2 + (palette[3 * j + 2] - b) ** 2;
                if (d < best) {
                    best = d;
                    idx = j;
                }
            }
            cache.set(key, idx);
        }
        out[i] = idx;
    }
    return out;
}

/**
 * An animated GIF: every frame the whole image, over one global palette, shown for its delay and looped.
 *
 * @param {{width: number, height: number, palette: Uint8Array, frames: {indices: Uint8Array, delay: number}[],
 *     loop?: number}} spec  width, height in pixels (1 to 65535); palette RGB triples (1 to 256 colors); each
 *     frame's indices into it, width·height of them, and its delay in seconds (kept to 0.01 s, GIF's unit);
 *     loop: times to repeat, 0 (default) forever
 * @returns {Uint8Array}  the file
 */
export function encodeGif({ width, height, palette, frames, loop = 0 }) {
    const colors = palette.length / 3;
    if (!(Number.isInteger(width) && Number.isInteger(height) && width >= 1 && height >= 1 && width <= 65535
        && height <= 65535))
        throw new RangeError('encodeGif: width and height from 1 to 65535 required');
    if (!(colors >= 1 && colors <= 256 && Number.isInteger(colors)))
        throw new RangeError('encodeGif: a palette of 1 to 256 RGB colors required');
    if (!frames.length)
        throw new RangeError('encodeGif: at least one frame required');
    // the table holds 2^(bits) colors, bits from 1 to 8; LZW starts from at least 2 bits (GIF's minimum code size)
    let bits = 1;
    while ((1 << bits) < colors)
        bits++;
    const out = byteWriter();
    out.text('GIF89a');
    out.u16(width);
    out.u16(height);
    out.byte(0x80 | ((bits - 1) << 4) | (bits - 1)); // global table, its color resolution and size
    out.byte(0); // background color index
    out.byte(0); // pixel aspect ratio: not given
    for (let j = 0; j < 1 << bits; j++) {
        for (let ch = 0; ch < 3; ch++)
            out.byte(j < colors ? palette[3 * j + ch] : 0);
    }
    out.bytes([0x21, 0xff, 0x0b]);
    out.text('NETSCAPE2.0');
    out.bytes([0x03, 0x01]);
    out.u16(loop);
    out.byte(0);
    const minCode = Math.max(2, bits);
    for (const frame of frames) {
        if (frame.indices.length !== width * height)
            throw new RangeError('encodeGif: each frame needs width·height indices');
        out.bytes([0x21, 0xf9, 0x04, DISPOSE_NONE << 2]);
        out.u16(Math.max(0, Math.round(frame.delay * 100)));
        out.bytes([0, 0]); // no transparent color; block end
        out.byte(0x2c);
        out.u16(0);
        out.u16(0);
        out.u16(width);
        out.u16(height);
        out.byte(0); // no local table, not interlaced
        out.byte(minCode);
        const data = lzw(frame.indices, minCode);
        for (let p = 0; p < data.length; p += 255) {
            const n = Math.min(255, data.length - p);
            out.byte(n);
            out.bytes(data.subarray(p, p + n));
        }
        out.byte(0);
    }
    out.byte(0x3b);
    return out.result();
}

/**
 * GIF's LZW: codes of minCode + 1 bits growing to 12, packed least significant bit first; a clear code when the
 * table is full. The table is keyed by (prefix code, next index) in typed arrays stamped with a generation, so a
 * clear costs nothing. The code-size rule (grow before adding the code that needs the extra bit) is the one GIF
 * decoders expect, one step ahead of the decoder's table.
 */
function lzw(indices, minCode) {
    const clear = 1 << minCode;
    const eoi = clear + 1;
    const table = new Int16Array(MAX_CODES * 256);
    const stamp = new Uint32Array(MAX_CODES * 256);
    let generation = 1;
    const out = byteWriter();
    let acc = 0;
    let accBits = 0;
    let size = minCode + 1;
    let next = eoi + 1;
    const emit = code => {
        acc |= code << accBits;
        accBits += size;
        while (accBits >= 8) {
            out.byte(acc & 255);
            acc >>>= 8;
            accBits -= 8;
        }
    };
    emit(clear);
    let prefix = indices[0];
    for (let i = 1; i < indices.length; i++) {
        const k = indices[i];
        const key = prefix * 256 + k;
        if (stamp[key] === generation) {
            prefix = table[key];
            continue;
        }
        emit(prefix);
        if (next === MAX_CODES) {
            emit(clear);
            generation++;
            size = minCode + 1;
            next = eoi + 1;
        }
        else {
            if (next >= 1 << size)
                size++;
            table[key] = next++;
            stamp[key] = generation;
        }
        prefix = k;
    }
    emit(prefix);
    emit(eoi);
    if (accBits > 0)
        out.byte(acc & 255);
    return out.result();
}

function byteWriter() {
    let buf = new Uint8Array(1 << 16);
    let n = 0;
    const room = extra => {
        if (n + extra <= buf.length)
            return;
        let size = buf.length * 2;
        while (size < n + extra)
            size *= 2;
        const grown = new Uint8Array(size);
        grown.set(buf.subarray(0, n));
        buf = grown;
    };
    return {
        byte(v) {
            room(1);
            buf[n++] = v;
        },
        bytes(arr) {
            room(arr.length);
            buf.set(arr, n);
            n += arr.length;
        },
        u16(v) {
            room(2);
            buf[n++] = v & 255;
            buf[n++] = (v >> 8) & 255;
        },
        text(s) {
            room(s.length);
            for (let i = 0; i < s.length; i++)
                buf[n++] = s.charCodeAt(i);
        },
        result: () => buf.slice(0, n),
    };
}
