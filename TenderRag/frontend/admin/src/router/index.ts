import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/login', name: 'Login', component: () => import('../views/Login.vue') },
  {
    path: '/',
    component: () => import('../layouts/MainLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      { path: '', redirect: '/documents' },
      { path: 'documents', name: 'Documents', component: () => import('../views/Documents.vue') },
      { path: 'data', name: 'DataBrowse', component: () => import('../views/DataBrowse.vue') },
      { path: 'chat', name: 'Chat', component: () => import('../views/Chat.vue') },
      { path: 'tasks', name: 'Tasks', component: () => import('../views/Tasks.vue') },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

const router = createRouter({ history: createWebHistory(), routes })

router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) {
    return next('/login')
  }
  if (token && to.name === 'Login') {
    return next('/')
  }
  next()
})

export default router