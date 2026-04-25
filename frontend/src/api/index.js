import axios from 'axios'
import { message } from 'ant-design-vue'

const BASE_URL = '/api'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('su_api_key')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.detail || err.message || '请求失败'
    message.error(msg)
    return Promise.reject(err)
  }
)

export const login = (params) => api.post('/tenant/create', params)

export const health = () => api.get('/health')

export const addMemory = (params) => api.post('/memory/add', params)

export const queryMemory = (params) => api.post('/memory/query', params)

export const deleteMemory = (params) => api.post('/memory/delete', params)

export const chat = (params) => api.post('/chat/completions', params)

export default api
