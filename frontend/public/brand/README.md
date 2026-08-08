# Brand assets

Exported logo files for places that can't take the app's SVG — AdSense's consent-message
branding, store listings, social cards. The canonical mark lives in `../logo.svg`
(and `../../app/icon.svg` for the favicon); regenerate these when that changes.

| File | Use |
| --- | --- |
| `brawldraft-logo-5x1-lighttheme.*` | Horizontal lockup, dark navy wordmark — for light backgrounds. **The AdSense upload.** |
| `brawldraft-logo-5x1-darktheme.*` | Same lockup, white wordmark — for dark backgrounds. |
| `brawldraft-logo-1024.png` | Square mark alone, 1024×1024. |

The lockups are exactly 5:1 (2560×512), AdSense's recommended ratio, and every PNG is well
under its 150 KB cap. PNGs are transparent, which is why the wordmark ships in two colors
rather than one: the site's blue-pink gradient wordmark disappears on light backgrounds, so
these use solid fills instead.

The mark is inset from the canvas edge on purpose — containers that crop or round the logo
would otherwise clip the tile.

## Regenerating

The `.svg` files next to each PNG are the sources. Rasterize with headless Chrome (the mark
is flat vector, so any renderer works, but Chrome matches what the browser shows):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --hide-scrollbars --default-background-color=00000000 --window-size=2560,512 \
  --screenshot=out.png page.html
```

where `page.html` is the `.svg` wrapped in `<body style="margin:0">`. Chrome screenshots a
page, not a file, so the wrapper (and the zeroed margin) is what makes the output land at the
exact pixel dimensions.
