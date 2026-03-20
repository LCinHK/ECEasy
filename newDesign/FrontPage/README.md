# ECEasy FrontPage

Initial static home page for ECEasy demo and introduction.

## Files

- `index.html` - main landing page
- `styles/main.css` - page styling
- `scripts/main.js` - small progressive enhancement script

## Use in local server

If your backend serves static files, point users to this page path directly:

- `/frontpage/index.html` (recommended mounted route)
- `/frontpage` (redirects to the same page)

To make this page the default homepage, set in `.env`:

```env
UI_VERSION=frontpage
```

Current call-to-action buttons target:

- `/newUI/index.html`
- `/ui/index.html`

## Preview quickly

```powershell
Push-Location "C:\Users\Lion Chen\OneDrive - HKUST Connect\FYP\ECEasy\newDesign\FrontPage"
python -m http.server 5500
Pop-Location
```

Open `http://localhost:5500` in your browser.

