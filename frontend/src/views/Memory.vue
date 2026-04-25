<template>
  <div class="memory-page">
    <!-- 筛选栏 -->
    <div class="filter-bar">
      <a-input
        v-model:value="filters.userId"
        placeholder="用户ID"
        style="width: 140px"
        allow-clear
      />
      <a-input
        v-model:value="filters.tenantId"
        placeholder="租户ID"
        style="width: 180px"
        allow-clear
      />
      <a-select
        v-model:value="filters.type"
        placeholder="记忆类型"
        style="width: 120px"
        allow-clear
      >
        <a-select-option value="fact">事实</a-select-option>
        <a-select-option value="preference">偏好</a-select-option>
        <a-select-option value="event">事件</a-select-option>
        <a-select-option value="belief">信念</a-select-option>
      </a-select>
      <a-range-picker
        v-model:value="filters.dateRange"
        @change="doFilter"
      />
      <a-input
        v-model:value="filters.keyword"
        placeholder="搜索关键词"
        allow-clear
        @pressEnter="doFilter"
        style="width: 160px"
      />
      <a-button type="primary" @click="doFilter">查询</a-button>
      <a-button @click="resetFilter">重置</a-button>
    </div>

    <!-- 统计卡片 -->
    <div class="stat-row">
      <div class="mini-stat">
        <span class="label">总计</span>
        <span class="value">{{ pagination.total }}</span>
      </div>
      <div class="mini-stat">
        <span class="label">事实</span>
        <span class="value blue">{{ typeStats.fact }}</span>
      </div>
      <div class="mini-stat">
        <span class="label">偏好</span>
        <span class="value pink">{{ typeStats.preference }}</span>
      </div>
      <div class="mini-stat">
        <span class="label">事件</span>
        <span class="value orange">{{ typeStats.event }}</span>
      </div>
      <div class="mini-stat">
        <span class="label">信念</span>
        <span class="value purple">{{ typeStats.belief }}</span>
      </div>
      <div class="mini-stat">
        <span class="label">已归档</span>
        <span class="value gray">{{ typeStats.archived }}</span>
      </div>
    </div>

    <!-- 记忆表格 -->
    <div class="table-card">
      <a-table
        :columns="columns"
        :data-source="memories"
        :loading="loading"
        :pagination="paginationConfig"
        @change="handleTableChange"
        row-key="id"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'content'">
            <div class="table-content">
              {{ record.content?.substring(0, 60) }}{{ record.content?.length > 60 ? '...' : '' }}
            </div>
          </template>
          <template v-else-if="column.key === 'type'">
            <a-tag :color="typeColor(record.metadata?.type)">
              {{ record.metadata?.type || 'fact' }}
            </a-tag>
          </template>
          <template v-else-if="column.key === 'status'">
            <a-tag :color="statusColor(record.status)">
              {{ record.status || 'active' }}
            </a-tag>
          </template>
          <template v-else-if="column.key === 'time'">
            {{ formatTime(record.timestamp) }}
          </template>
          <template v-else-if="column.key === 'actions'">
            <a-space>
              <a-button size="small" type="link" @click="showDetail(record)">
                详情
              </a-button>
              <a-popconfirm
                title="确认删除？"
                ok-text="删除"
                cancel-text="取消"
                @confirm="doDelete(record.id)"
              >
                <a-button size="small" type="link" danger>删除</a-button>
              </a-popconfirm>
              <a-button size="small" type="link" @click="doArchive(record)">
                归档
              </a-button>
            </a-space>
          </template>
        </template>
      </a-table>
    </div>

    <!-- 详情弹窗 -->
    <a-modal
      v-model:open="detailVisible"
      title="记忆详情"
      width="640px"
      :footer="null"
    >
      <div v-if="currentRecord" class="detail-content">
        <div class="detail-row">
          <span class="detail-label">记忆ID</span>
          <span class="detail-value">{{ currentRecord.id }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">用户/租户</span>
          <span class="detail-value">{{ currentRecord.metadata?.user_id }} / {{ currentRecord.tenantId }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">类型</span>
          <a-tag :color="typeColor(currentRecord.metadata?.type)">
            {{ currentRecord.metadata?.type || 'fact' }}
          </a-tag>
        </div>
        <div class="detail-row">
          <span class="detail-label">状态</span>
          <a-tag :color="statusColor(currentRecord.status)">
            {{ currentRecord.status || 'active' }}
          </a-tag>
        </div>
        <div class="detail-row">
          <span class="detail-label">时间</span>
          <span class="detail-value">{{ formatTime(currentRecord.timestamp) }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">置信度</span>
          <span class="detail-value">{{ ((currentRecord.metadata?.confidence || 0.5) * 100).toFixed(0) }}%</span>
        </div>
        <div class="detail-section-title">原始内容</div>
        <div class="detail-raw">{{ currentRecord.content }}</div>
        <div class="detail-section-title">结构化内容</div>
        <div class="detail-structured">
          <div>类型: {{ currentRecord.metadata?.type || 'fact' }}</div>
          <div>优先级: {{ currentRecord.metadata?.priority || 5 }}</div>
          <div>来源: {{ currentRecord.metadata?.source || '用户输入' }}</div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import { queryMemory, deleteMemory } from '@/api'

const loading = ref(false)
const memories = ref([])
const detailVisible = ref(false)
const currentRecord = ref(null)

const filters = reactive({
  userId: '',
  tenantId: '',
  type: undefined,
  dateRange: null,
  keyword: ''
})

const pagination = reactive({ total: 0, current: 1, pageSize: 20 })
const paginationConfig = computed(() => ({
  current: pagination.current,
  pageSize: pagination.pageSize,
  total: pagination.total,
  showSizeChanger: true,
  showTotal: (t) => `共 ${t} 条`
}))

const typeStats = computed(() => {
  const stats = { fact: 0, preference: 0, event: 0, belief: 0, archived: 0 }
  memories.value.forEach(m => {
    const t = m.metadata?.type || 'fact'
    if (m.status === 'archived') stats.archived++
    else if (stats[t] !== undefined) stats[t]++
  })
  return stats
})

const columns = [
  { title: 'ID', key: 'id', width: 80, ellipsis: true },
  { title: '用户/租户', key: 'user', width: 160, ellipsis: true },
  { title: '内容摘要', key: 'content' },
  { title: '类型', key: 'type', width: 90 },
  { title: '状态', key: 'status', width: 90 },
  { title: '时间', key: 'time', width: 160 },
  { title: '操作', key: 'actions', width: 160, fixed: 'right' }
]

const typeColor = (t) => ({ fact: 'blue', preference: 'pink', event: 'orange', belief: 'purple' }[t] || 'blue')
const statusColor = (s) => ({ active: 'green', archived: 'default', merged: 'orange', conflict: 'red' }[s] || 'green')
const formatTime = (ts) => ts ? dayjs(ts * 1000).format('YYYY-MM-DD HH:mm') : '-'

async function loadData() {
  loading.value = true
  try {
    const r = await queryMemory({
      user_id: filters.userId || 'all',
      query: filters.keyword || '',
      limit: pagination.pageSize
    })
    memories.value = r.memories || []
    pagination.total = memories.value.length
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

function doFilter() {
  pagination.current = 1
  loadData()
}

function resetFilter() {
  Object.assign(filters, { userId: '', tenantId: '', type: undefined, dateRange: null, keyword: '' })
  doFilter()
}

function handleTableChange(pag) {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  loadData()
}

function showDetail(record) {
  currentRecord.value = record
  detailVisible.value = true
}

async function doDelete(id) {
  try {
    await deleteMemory({ user_id: 'demo', memory_id: id })
    message.success('删除成功')
    loadData()
  } catch (e) {
    message.error('删除失败')
  }
}

function doArchive(record) {
  message.info('归档功能开发中')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.memory-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.filter-bar {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  background: #1e1e1e;
  padding: 16px;
  border-radius: 8px;
}
.stat-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.mini-stat {
  background: #1e1e1e;
  padding: 8px 16px;
  border-radius: 6px;
  display: flex;
  gap: 8px;
  align-items: center;
}
.mini-stat .label {
  font-size: 13px;
  color: #666;
}
.mini-stat .value {
  font-size: 16px;
  font-weight: 600;
}
.value.blue { color: #6366f1; }
.value.pink { color: #ec4899; }
.value.orange { color: #f97316; }
.value.purple { color: #a855f7; }
.value.gray { color: #666; }
.table-card {
  background: #1e1e1e;
  border-radius: 8px;
  padding: 16px;
}
.table-content {
  font-size: 13px;
  color: #aaa;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.detail-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.detail-row {
  display: flex;
  gap: 16px;
  align-items: center;
}
.detail-label {
  width: 80px;
  color: #666;
  font-size: 13px;
  flex-shrink: 0;
}
.detail-value {
  color: #ccc;
  font-size: 13px;
}
.detail-section-title {
  font-size: 13px;
  color: #666;
  margin-top: 8px;
  border-top: 1px solid #333;
  padding-top: 8px;
}
.detail-raw {
  background: #141414;
  padding: 12px;
  border-radius: 6px;
  font-size: 13px;
  color: #aaa;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}
.detail-structured {
  background: #141414;
  padding: 12px;
  border-radius: 6px;
  font-size: 13px;
  color: #aaa;
  line-height: 1.8;
}
</style>
