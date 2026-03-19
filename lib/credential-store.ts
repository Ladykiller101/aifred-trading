// Client-side broker credential storage
// Credentials are obfuscated in localStorage to prevent casual reading.
// For production, consider using the Web Crypto API with a user-derived key.

const STORAGE_KEY = 'aifred_broker_credentials';
const OBFUSCATION_KEY = 'AIFr3d-Tr4d1ng-2026';

export interface StoredBrokerCredentials {
  [brokerId: string]: {
    credentials: Record<string, string>;
    connectedAt: string;
    lastTested: string;
    testResult: 'success' | 'failed';
    accountInfo?: {
      balance: Record<string, number>;
      accountId: string;
    };
  };
}

function obfuscate(text: string): string {
  const encoded = Array.from(text).map((char, i) =>
    String.fromCharCode(char.charCodeAt(0) ^ OBFUSCATION_KEY.charCodeAt(i % OBFUSCATION_KEY.length))
  ).join('');
  return btoa(encoded);
}

function deobfuscate(encoded: string): string {
  const decoded = atob(encoded);
  return Array.from(decoded).map((char, i) =>
    String.fromCharCode(char.charCodeAt(0) ^ OBFUSCATION_KEY.charCodeAt(i % OBFUSCATION_KEY.length))
  ).join('');
}

export function saveCredentials(
  brokerId: string,
  credentials: Record<string, string>,
  testResult: { success: boolean; balance?: Record<string, number>; accountId?: string },
) {
  const stored = loadAllCredentials();
  stored[brokerId] = {
    credentials,
    connectedAt: new Date().toISOString(),
    lastTested: new Date().toISOString(),
    testResult: testResult.success ? 'success' : 'failed',
    accountInfo: testResult.success ? {
      balance: testResult.balance || {},
      accountId: testResult.accountId || `${brokerId}_${Date.now().toString(36)}`,
    } : undefined,
  };
  localStorage.setItem(STORAGE_KEY, obfuscate(JSON.stringify(stored)));
}

export function loadAllCredentials(): StoredBrokerCredentials {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(deobfuscate(raw));
  } catch {
    return {};
  }
}

export function loadCredentials(brokerId: string): Record<string, string> | null {
  const stored = loadAllCredentials();
  return stored[brokerId]?.credentials || null;
}

export function removeCredentials(brokerId: string) {
  const stored = loadAllCredentials();
  delete stored[brokerId];
  if (Object.keys(stored).length === 0) {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, obfuscate(JSON.stringify(stored)));
  }
}

export function isConnected(brokerId: string): boolean {
  const stored = loadAllCredentials();
  return stored[brokerId]?.testResult === 'success';
}

export function getConnectedBrokerIds(): string[] {
  const stored = loadAllCredentials();
  return Object.keys(stored).filter(id => stored[id].testResult === 'success');
}
