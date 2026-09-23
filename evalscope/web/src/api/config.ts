export interface AppConfig {
  outputs_root?: string
  server_address?: string
  registration_mode?: 'admin_only' | 'invite' | 'public'
  registration_mode_locked?: boolean
}

type FetchConfig = () => Promise<Response>

export function createConfigLoader(fetchConfig: FetchConfig): () => Promise<AppConfig> {
  let request: Promise<AppConfig> | null = null

  return () => {
    if (!request) {
      request = fetchConfig()
        .then(async (response) => {
          if (!response.ok) throw new Error('Failed to load application config')
          return response.json() as Promise<AppConfig>
        })
        .catch((error) => {
          request = null
          throw error
        })
    }
    return request
  }
}

export const loadAppConfig = createConfigLoader(() => fetch('/api/v1/config'))
