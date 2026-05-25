<script setup lang="ts">
import { ref, onMounted } from 'vue'
import api from '../api'

const activeTab = ref('tasks')

// Index Tasks
const tasks = ref<any[]>([])
const tasksPage = ref(1)
const tasksPageSize = ref(20)

const taskStatusMap: Record<string, string> = {
  queued: 'info',
  running: 'warning',
  done: 'success',
  failed: 'danger',
}
const taskStatusText: Record<string, string> = {
  queued: '排队中',
  running: '处理中',
  done: '已完成',
  failed: '失败',
}

async function fetchTasks() {
  const { data } = await api.get('/api/index/tasks', {
    params: { page: tasksPage.value, page_size: tasksPageSize.value },
  })
  tasks.value = data
}

// Intent Logs
const logs = ref<any[]>([])
const logsPage = ref(1)
const logsPageSize = ref(50)
const failedLevel = ref('')

async function fetchLogs() {
  const params: any = { page: logsPage.value, page_size: logsPageSize.value }
  if (failedLevel.value) params.failed_level = failedLevel.value
  const { data } = await api.get('/api/intent-logs', { params })
  logs.value = data
}

onMounted(() => {
  fetchTasks()
  fetchLogs()
})
</script>

<template>
  <div class="tasks-page">
    <h2>任务 & 日志</h2>

    <el-tabs v-model="activeTab">
      <!-- Index Tasks Tab -->
      <el-tab-pane label="索引任务" name="tasks">
        <el-table :data="tasks" stripe style="width: 100%">
          <el-table-column prop="id" label="ID" width="80" />
          <el-table-column prop="task_type" label="类型" width="80">
            <template #default="{ row }">
              <el-tag size="small" :type="row.task_type === 'full' ? 'primary' : 'warning'">
                {{ row.task_type === 'full' ? '全量' : '增量' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="category" label="类别" width="80" />
          <el-table-column prop="status" label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="taskStatusMap[row.status] || 'info'" size="small">
                {{ taskStatusText[row.status] || row.status || 'queued' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="result_msg" label="结果" min-width="200" show-overflow-tooltip />
          <el-table-column prop="created_at" label="创建时间" width="180" />
        </el-table>
        <div class="pagination-wrap">
          <el-pagination
            v-model:current-page="tasksPage"
            :page-size="tasksPageSize"
            layout="prev, pager, next"
            @current-change="fetchTasks"
          />
        </div>
      </el-tab-pane>

      <!-- Intent Logs Tab -->
      <el-tab-pane label="意图日志" name="logs">
        <div class="log-filter">
          <el-select v-model="failedLevel" placeholder="全部级别" clearable @change="fetchLogs" style="width: 150px">
            <el-option label="全部" value="" />
            <el-option label="L1 - Jieba" value="L1" />
            <el-option label="L2 - BERT" value="L2" />
            <el-option label="L3 - LLM" value="L3" />
          </el-select>
        </div>
        <el-table :data="logs" stripe style="width: 100%">
          <el-table-column prop="id" label="ID" width="80" />
          <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
          <el-table-column prop="failed_level" label="未命中级别" width="120">
            <template #default="{ row }">
              <el-tag size="small">{{ row.failed_level }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="final_intent" label="最终意图" width="120" />
          <el-table-column prop="created_at" label="时间" width="180" />
        </el-table>
        <div class="pagination-wrap">
          <el-pagination
            v-model:current-page="logsPage"
            :page-size="logsPageSize"
            layout="prev, pager, next"
            @current-change="fetchLogs"
          />
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.tasks-page {
  background: #fff;
  padding: 20px;
  border-radius: 4px;
}
.tasks-page h2 {
  margin: 0 0 20px;
}

.log-filter {
  margin-bottom: 16px;
}

.pagination-wrap {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}
</style>