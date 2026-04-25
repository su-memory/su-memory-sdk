import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/console' },
  { path: '/console', name: 'console', component: () => import('../views/Console.vue') },
  { path: '/memory', name: 'memory', component: () => import('../views/Memory.vue') },
  { path: '/admin', name: 'admin', component: () => import('../views/Admin.vue') },
]

export default createRouter({
  history: createWebHistory(),
  routes
})
