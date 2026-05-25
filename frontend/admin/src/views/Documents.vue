<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const docs = ref<any[]>([])
const category = ref('')
const page = ref(1)
const pageSize = ref(20)
const total = ref(0)
const uploadCategory = ref('legal')
const uploading = ref(false)
const editingDoc = ref<any>(null)
const editDialog = ref(false)
const editForm = ref({ filename: '', category: '' })

const statusMap: Record<string, string> = {
  pending: 'info',
  processing: 'warning',
  done: 'success',
  failed: 'danger',
}
const statusText: Record<string, string> = {
  pending: '待处理',
  processing: '处理中',
  done: '已完成',
  failed: '失败',
}

async function fetchDocs() {
  const { data } = await api.get('/api/documents', { params: { category: category.value || undefined, page: page.value, page_size: pageSize.value } })
  docs.value = data
  total.value = data.length >= pageSize.value ? (page.value * pageSize.value + 1) : page.value * pageSize.value
}

async function handleUpload(options: any) {
  uploading.value = true
  try {
    const rawFile = options.file?.raw || options.file
    const formData = new FormData()
    formData.append('file', rawFile)
    await api.post('/api/documents/upload', formData, {
      params: { category: uploadCategory.value },
    })
    ElMessage.success('上传成功')
    fetchDocs()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '上传失败')
  } finally {
    uploading.value = false
  }
  return false
}

function openEdit(doc: any) {
  editingDoc.value = doc
  editForm.value = { filename: doc.filename, category: doc.category }
  editDialog.value = true
}

async function saveEdit() {
  if (!editingDoc.value) return
  try {
    await api.put(`/api/documents/${editingDoc.value.id}`, editForm.value)
    ElMessage.success('修改成功')
    editDialog.value = false
    fetchDocs()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '修改失败')
  }
}

async function deleteDoc(doc: any) {
  try {
    await ElMessageBox.confirm(`确定删除 "${doc.filename}"？`, '确认删除', { type: 'warning' })
    await api.delete(`/api/documents/${doc.id}`)
    ElMessage.success('删除成功')
    fetchDocs()
  } catch {
    // cancelled
  }
}

onMounted(fetchDocs)
</script>

<template>
  <div class="documents-page">
    <div class="page-header">
      <h2>文档管理</h2>
      <div class="header-actions">
        <el-select v-model="category" placeholder="全部分类" clearable @change="fetchDocs" style="width: 150px">
          <el-option label="全部" value="" />
          <el-option label="法律" value="legal" />
          <el-option label="招标" value="tender" />
          <el-option label="产品" value="product" />
        </el-select>
        <el-select v-model="uploadCategory" style="width: 120px">
          <el-option label="法律" value="legal" />
          <el-option label="招标" value="tender" />
          <el-option label="产品" value="product" />
        </el-select>
        <el-upload
          :show-file-list="false"
          :http-request="handleUpload"
          :disabled="uploading"
        >
          <el-button type="primary" :loading="uploading">上传文件</el-button>
        </el-upload>
      </div>
    </div>

    <el-table :data="docs" stripe style="width: 100%">
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="filename" label="文件名" min-width="200" />
      <el-table-column prop="category" label="类别" width="80" />
      <el-table-column prop="file_size" label="大小" width="100">
        <template #default="{ row }">
          {{ row.file_size ? (row.file_size / 1024).toFixed(1) + ' KB' : '-' }}
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="statusMap[row.status] || 'info'" size="small">
            {{ statusText[row.status] || row.status || 'pending' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="上传时间" width="180" />
      <el-table-column label="操作" width="160">
        <template #default="{ row }">
          <el-button size="small" @click="openEdit(row)">编辑</el-button>
          <el-button size="small" type="danger" @click="deleteDoc(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="pagination-wrap">
      <el-pagination
        v-model:current-page="page"
        :page-size="pageSize"
        layout="prev, pager, next"
        @current-change="fetchDocs"
      />
    </div>

    <!-- Edit Dialog -->
    <el-dialog v-model="editDialog" title="编辑文档" width="400px">
      <el-form>
        <el-form-item label="文件名">
          <el-input v-model="editForm.filename" />
        </el-form-item>
        <el-form-item label="类别">
          <el-select v-model="editForm.category" style="width: 100%">
            <el-option label="法律" value="legal" />
            <el-option label="招标" value="tender" />
            <el-option label="产品" value="product" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editDialog = false">取消</el-button>
        <el-button type="primary" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.documents-page {
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
  align-items: center;
}

.pagination-wrap {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}
</style>
