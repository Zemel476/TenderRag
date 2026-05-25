<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { useUserStore } from '../stores/user'
import api from '../api'

const userStore = useUserStore()

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const sessionId = ref<string | null>(null)
const messages = ref<Message[]>([])
const inputText = ref('')
const sending = ref(false)
const messagesEl = ref<HTMLElement | null>(null)

async function ensureSession() {
  if (sessionId.value) return
  const { data } = await api.post('/api/sessions', { title: '内部问答' })
  sessionId.value = String(data.session_id)
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || sending.value) return
  await ensureSession()

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  sending.value = true

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
      body: JSON.stringify({ session_id: sessionId.value, message: text }),
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
            }
          } catch { /* partial JSON, ignore */ }
        }
      }
    }
  } catch (e: any) {
    messages.value[assistantIdx].content = '抱歉，请求失败：' + (e.message || '未知错误')
  } finally {
    sending.value = false
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

onMounted(() => {
  ensureSession()
})
</script>

<template>
  <div class="chat-page">
    <div class="chat-messages" ref="messagesEl">
      <div v-for="(msg, idx) in messages" :key="idx" class="message-row" :class="msg.role">
        <div class="message-bubble" :class="msg.role">
          <div class="message-content">{{ msg.content }}</div>
        </div>
      </div>
      <div v-if="messages.length === 0" class="empty-chat">
        <h3>内部问答</h3>
        <p>输入问题进行内部知识检索</p>
      </div>
    </div>

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
  </div>
</template>

<style scoped>
.chat-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 100px);
  background: #fff;
  border-radius: 4px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
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
  padding: 16px 20px;
  border-top: 1px solid #e4e7ed;
  align-items: flex-end;
}
.chat-input-area .el-textarea {
  flex: 1;
}
</style>