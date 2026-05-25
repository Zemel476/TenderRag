<script setup lang="ts">
import { ref, nextTick, watch, onMounted } from 'vue'
import { useUserStore } from '../stores/user'
import api from '../api'

const userStore = useUserStore()

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const sessions = ref<{ id: number; title: string }[]>([])
const activeSessionId = ref<number | null>(null)
const messages = ref<Message[]>([])
const inputText = ref('')
const sending = ref(false)
const messagesEl = ref<HTMLElement | null>(null)

async function fetchSessions() {
  const { data } = await api.get('/api/sessions')
  sessions.value = data
}

async function createSession() {
  const { data } = await api.post('/api/sessions', { title: '新对话' })
  const session = { id: data.session_id, title: data.title }
  sessions.value.unshift(session)
  activeSessionId.value = data.session_id
  messages.value = []
}

async function selectSession(id: number) {
  activeSessionId.value = id
  const { data } = await api.get(`/api/sessions/${id}/messages`)
  messages.value = data.map((m: any) => ({
    role: m.role,
    content: m.content,
  }))
  await scrollToBottom()
}

async function deleteSession(id: number) {
  await api.delete(`/api/sessions/${id}`)
  sessions.value = sessions.value.filter((s) => s.id !== id)
  if (activeSessionId.value === id) {
    activeSessionId.value = null
    messages.value = []
  }
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || sending.value) return
  if (!activeSessionId.value) await createSession()

  const sid = String(activeSessionId.value)
  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  sending.value = true

  // Add placeholder for assistant response
  messages.value.push({ role: 'assistant', content: '' })
  const assistantIdx = messages.value.length - 1
  await scrollToBottom()

  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${userStore.token}`,
      },
      body: JSON.stringify({ session_id: sid, message: text }),
    })

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) throw new Error('No reader')

    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const json = line.slice(6)
          try {
            const event = JSON.parse(json)
            if (event.type === 'message') {
              messages.value[assistantIdx].content += event.content
              await scrollToBottom()
            } else if (event.type === 'done') {
              break
            }
          } catch {
            // Partial JSON, ignore
          }
        }
      }
    }
  } catch (e: any) {
    messages.value[assistantIdx].content = '抱歉，请求失败：' + (e.message || '未知错误')
  } finally {
    sending.value = false
    // Refresh sessions list in case title changed
    fetchSessions()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

function formatTime() {
  const now = new Date()
  return now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function isNewSession() {
  return !activeSessionId.value
}

function logout() {
  userStore.logout()
  window.location.href = '/login'
}

onMounted(() => {
  fetchSessions()
})
</script>

<template>
  <div class="chat-layout">
    <!-- Left Sidebar -->
    <aside class="chat-sidebar">
      <div class="sidebar-header">
        <h3>对话列表</h3>
        <el-button type="primary" size="small" @click="createSession" :disabled="isNewSession()">
          + 新对话
        </el-button>
      </div>
      <div class="session-list" v-if="sessions.length > 0">
        <div
          v-for="s in sessions"
          :key="s.id"
          class="session-item"
          :class="{ active: activeSessionId === s.id }"
          @click="selectSession(s.id)"
        >
          <span class="session-title">{{ s.title || '新对话' }}</span>
          <el-button
            type="danger"
            size="small"
            :icon="'Delete'"
            text
            @click.stop="deleteSession(s.id)"
          />
        </div>
      </div>
      <div v-else class="empty-sessions">
        <p>暂无对话</p>
        <el-button type="primary" @click="createSession">开始新对话</el-button>
      </div>
    </aside>

    <!-- Right Chat Area -->
    <main class="chat-main">
      <header class="chat-header">
        <div class="header-left">
          <h2>招标智能问答</h2>
        </div>
        <div class="header-right">
          <span v-if="userStore.user">{{ userStore.user.username }}</span>
          <el-button text @click="logout">退出</el-button>
        </div>
      </header>

      <!-- Messages -->
      <div class="chat-messages" ref="messagesEl">
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          class="message-row"
          :class="msg.role"
        >
          <div class="message-bubble" :class="msg.role">
            <div class="message-content">{{ msg.content }}</div>
            <div class="message-time">{{ formatTime() }}</div>
          </div>
        </div>
        <div v-if="messages.length === 0 && !sending" class="empty-chat">
          <h3>欢迎使用招标智能问答</h3>
          <p>选择或创建对话开始提问</p>
        </div>
      </div>

      <!-- Input Area -->
      <div class="chat-input-area">
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="输入问题，按 Enter 发送，Shift+Enter 换行"
          @keydown="handleKeydown"
          :disabled="sending"
        />
        <el-button
          type="primary"
          @click="sendMessage"
          :loading="sending"
          :disabled="!inputText.trim() || sending"
        >
          发送
        </el-button>
      </div>
    </main>
  </div>
</template>

<style scoped>
.chat-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.chat-sidebar {
  width: 280px;
  min-width: 280px;
  background: #f5f7fa;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #e4e7ed;
}
.sidebar-header h3 {
  margin: 0 0 12px;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}
.session-item:hover {
  background: #e4e7ed;
}
.session-item.active {
  background: #d9ecff;
  border: 1px solid #409eff;
}
.session-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
}

.empty-sessions {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: #909399;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}
.chat-header h2 {
  margin: 0;
  font-size: 18px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
}

.message-row {
  margin-bottom: 16px;
  display: flex;
}
.message-row.user {
  justify-content: flex-end;
}
.message-row.assistant {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.6;
}
.message-bubble.user {
  background: #409eff;
  color: #fff;
}
.message-bubble.assistant {
  background: #f0f2f5;
  color: #303133;
}
.message-content {
  white-space: pre-wrap;
  word-break: break-word;
}
.message-time {
  font-size: 11px;
  margin-top: 4px;
  opacity: 0.6;
  text-align: right;
}

.empty-chat {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.chat-input-area {
  display: flex;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid #e4e7ed;
  background: #fff;
  align-items: flex-end;
}
.chat-input-area .el-textarea {
  flex: 1;
}
</style>