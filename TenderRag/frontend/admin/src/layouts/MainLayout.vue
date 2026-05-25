<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from '../stores/user'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()

const activeTab = computed(() => {
  const path = route.path
  if (path.startsWith('/documents')) return '/documents'
  if (path.startsWith('/data')) return '/data'
  if (path.startsWith('/chat')) return '/chat'
  if (path.startsWith('/tasks')) return '/tasks'
  return '/documents'
})

function logout() {
  userStore.logout()
  router.push('/login')
}
</script>

<template>
  <el-container class="admin-layout">
    <el-header class="admin-header">
      <div class="header-left">
        <h3>招标RAG管理系统</h3>
        <el-menu
          mode="horizontal"
          :default-active="activeTab"
          router
          class="header-menu"
        >
          <el-menu-item index="/documents">文档管理</el-menu-item>
          <el-menu-item index="/data">数据浏览</el-menu-item>
          <el-menu-item index="/chat">内部问答</el-menu-item>
          <el-menu-item index="/tasks">任务日志</el-menu-item>
        </el-menu>
      </div>
      <div class="header-right">
        <span>{{ userStore.user?.username }}</span>
        <el-button text type="danger" @click="logout">退出</el-button>
      </div>
    </el-header>
    <el-main class="admin-main">
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
.admin-layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 24px;
}
.header-left h3 {
  margin: 0;
  white-space: nowrap;
}

.header-menu {
  border-bottom: none !important;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 14px;
}

.admin-main {
  flex: 1;
  padding: 20px;
  background: #f5f7fa;
  overflow-y: auto;
}
</style>