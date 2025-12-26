def follow_link(url: str, link_text: str):
    """
    Segui un link nella pagina web.
    """
    
    try:
        import playwright
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError("playwright is not installed. Please install it with 'pip install playwright'")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        page.get_by_role("link", name=link_text).click()
        return page.url


def snapshot_minimal(url, max_items=40):
    """
    Riduce il contesto per il modello: elementi interattivi principali
    (ruolo, nome/testo visibile, selettori risolvibili).
    """

    try:
        import playwright
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError("playwright is not installed. Please install it with 'pip install playwright'")


    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)

        flat = []

        # Preferisci selettori ARIA (role+name) quando possibile
        # Qui estraiamo una lista di elementi tipici e ne prendiamo testo e bounding box
        roles = ["button", "link", "textbox", "combobox", "checkbox"]
        for role in roles:
            try:
                locs = page.get_by_role(role).all()
            except Exception:
                locs = []
            for loc in locs[:max_items]:
                info = {"role": role, "name": None, "selector": None}
                try:
                    # prova a derivare un "selector" stabile: prima ARIA, poi CSS
                    # (Playwright può risolvere get_by_role(role, name=...) direttamente)
                    name = None
                    try:
                        name = loc.inner_text(timeout=500).strip()
                        if not name:
                            name = loc.get_attribute("aria-label")
                    except Exception:
                        pass
                    info["name"] = name or ""

                    # non c'è una API pubblica per estrarre il selettore: lo ricaviamo euristicamente
                    # come fallback, usiamo un CSS breve (potrebbe non essere unico)
                    try:
                        elt = loc.element_handle()
                        sel = elt.evaluate("""
                            (el) => {
                            // genera un CSS breve euristico
                            const id = el.id ? '#' + el.id : '';
                            const cls = (el.className && typeof el.className === 'string')
                                ? '.' + el.className.trim().split(/\\s+/).slice(0,2).join('.') : '';
                            return el.tagName.toLowerCase() + id + cls;
                            }
                        """)
                        info["selector"] = sel
                    except Exception:
                        info["selector"] = None
                except Exception:
                    continue
                flat.append(info)
                if len(flat) >= max_items:
                    break
            if len(flat) >= max_items:
                break

        # URL corrente come contesto
        ctx = {
            "url": page.url,
            "interactives": flat[:max_items]
        }
        return ctx


if __name__ == "__main__":
    #res = snapshot_minimal("https://profession.ai")
    res = follow_link("https://profession.ai", "Master Tech & AI")
    print(res)