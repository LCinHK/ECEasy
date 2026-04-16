interface MatomoInitOptions {
  baseUrl?: string;
  siteId?: string;
}

type MatomoWindow = Window & {
  _paq?: Array<unknown[]>;
  __eceasyMatomoInitialized__?: boolean;
};

const DEFAULT_BASE_URL = 'https://stats.alevel.tech/';
const DEFAULT_SITE_ID = '3';

function normalizeBaseUrl(rawUrl: string): string {
  return rawUrl.endsWith('/') ? rawUrl : `${rawUrl}/`;
}

function isMatomoEnabled(): boolean {
  const raw = import.meta.env.VITE_MATOMO_ENABLED;
  if (!raw) return true;
  return raw.toLowerCase() !== 'false';
}

export function initMatomoTracking(options: MatomoInitOptions = {}): void {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  if (!isMatomoEnabled()) return;

  const w = window as MatomoWindow;
  if (w.__eceasyMatomoInitialized__) return;

  const configuredBaseUrl = import.meta.env.VITE_MATOMO_URL || options.baseUrl || DEFAULT_BASE_URL;
  const siteId = import.meta.env.VITE_MATOMO_SITE_ID || options.siteId || DEFAULT_SITE_ID;
  const baseUrl = normalizeBaseUrl(configuredBaseUrl);

  const queue = (w._paq = w._paq || []);
  queue.push(['setTrackerUrl', `${baseUrl}matomo.php`]);
  queue.push(['setSiteId', siteId]);
  queue.push(['trackPageView']);
  queue.push(['enableLinkTracking']);

  const existingScript = document.querySelector<HTMLScriptElement>('script[data-eceasy-matomo="true"]');
  if (!existingScript) {
    const script = document.createElement('script');
    script.async = true;
    script.src = `${baseUrl}matomo.js`;
    script.setAttribute('data-eceasy-matomo', 'true');

    const firstScript = document.getElementsByTagName('script')[0];
    if (firstScript?.parentNode) {
      firstScript.parentNode.insertBefore(script, firstScript);
    } else {
      document.head.appendChild(script);
    }
  }

  w.__eceasyMatomoInitialized__ = true;
}

