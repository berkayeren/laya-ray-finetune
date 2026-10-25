// Same-origin JSON requests to the demo backend (demo.py on 127.0.0.1). Every call resolves to a discriminated
// result and never throws:
//   {ok: true, data}                        the server answered 2xx with JSON;
//   {ok: false, status: number, error}      the server answered an HTTP error; error is its one-line "error" field;
//   {ok: false, status: null, error: ""}    the server did not answer (unreachable).
// Paths are same-origin only ("/api/..."), so the page never reaches another host.

export type Result<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; error: string }
  | { ok: false; status: null; error: "" };

const UNREACHABLE = { ok: false, status: null, error: "" } as const;

async function toResult<T>(response: Response): Promise<Result<T>> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (response.ok) {
    return { ok: true, data: body as T };
  }
  const error =
    body !== null && typeof body === "object" && typeof (body as { error?: unknown }).error === "string"
      ? (body as { error: string }).error
      : "HTTP " + response.status;
  return { ok: false, status: response.status, error };
}

export async function getJSON<T>(path: string): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(path, { method: "GET", headers: { Accept: "application/json" } });
  } catch {
    return UNREACHABLE;
  }
  return toResult<T>(response);
}

export async function postJSON<T>(path: string, body: unknown): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    return UNREACHABLE;
  }
  return toResult<T>(response);
}
