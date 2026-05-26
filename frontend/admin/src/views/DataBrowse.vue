<script setup lang="ts">
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const category = ref('legal')
const rows = ref<any[]>([])
const columns = ref<string[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const selectedIds = ref<number[]>([])
const buildDialog = ref(false)
const taskType = ref('full')

async function fetchData() {
  try {
    const { data } = await api.get(`/api/data/${category.value}`, {
      params: { page: page.value, page_size: pageSize.value },
    })
    rows.value = data.rows || []
    total.value = data.total || 0
    if (rows.value.length > 0) {
      columns.value = Object.keys(rows.value[0]).filter((k) => k !== 'id')
    }
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '获取数据失败')
  }
}

watch(category, () => {
  page.value = 1
  fetchData()
})

async function handleBuildIndex() {
  const body: Record<string, unknown> = {
    task_type: taskType.value,
    category: category.value,
  }
  if (selectedIds.value.length > 0) {
    body.document_ids = selectedIds.value
  }
  try {
    await api.post('/api/index/build', body)
    ElMessage.success('索引构建任务已提交')
    buildDialog.value = false
  } catch (e: any) {
    const detail = e.response?.data?.detail
    if (Array.isArray(detail)) {
      const messages = detail.map((d: any) => d.msg || d.type).join('; ')
      ElMessage.error(messages || '提交失败')
    } else {
      ElMessage.error(detail || '提交失败')
    }
  }
}

function handleSelectionChange(selection: any[]) {
  selectedIds.value = selection.map((r: any) => r.id)
}

fetchData()
</script>

<template>
  <div class="data-browse-page">
    <div class="page-header">
      <h2>数据浏览</h2>
      <div class="header-actions">
        <el-select v-model="category" style="width: 150px">
          <el-option label="法律法规" value="legal" />
          <el-option label="政府招标信息" value="tender" />
          <el-option label="市场产品信息" value="product" />
        </el-select>
        <el-button type="primary" @click="buildDialog = true">构建索引</el-button>
      </div>
    </div>

    <el-table
      :data="rows"
      stripe
      style="width: 100%"
      max-height="calc(100vh - 240px)"
      @selection-change="handleSelectionChange"
    >
      <el-table-column type="selection" width="50" />
      <el-table-column
        v-for="col in columns"
        :key="col"
        :prop="col"
        :label="col"
        min-width="150"
        show-overflow-tooltip
      />
    </el-table>

    <div class="pagination-wrap">
      <el-pagination
        v-model:current-page="page"
        :page-size="pageSize"
        :total="total"
        layout="prev, pager, next, total"
        @current-change="fetchData"
      />
    </div>

    <!-- Build Index Dialog -->
    <el-dialog v-model="buildDialog" title="构建索引" width="400px">
      <el-form>
        <el-form-item label="任务类型">
          <el-radio-group v-model="taskType">
            <el-radio value="full">全量</el-radio>
            <el-radio value="incremental">增量</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="选中记录">
          <span>{{ selectedIds.length > 0 ? `已选择 ${selectedIds.length} 条` : '全量构建' }}</span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="buildDialog = false">取消</el-button>
        <el-button type="primary" @click="handleBuildIndex">确认</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.data-browse-page {
  background: #fff;
  padding: 20px;
  border-radius: 4px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.page-header h2 {
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.pagination-wrap {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}
</style>