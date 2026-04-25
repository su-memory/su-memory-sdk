<template>
  <div class="admin-page">
    <a-tabs v-model:activeKey="activeTab">
      <!-- 租户管理 -->
      <a-tab-pane key="tenants" tab="租户管理">
        <div class="section">
          <div class="section-header">
            <span class="section-title">租户列表</span>
            <a-button type="primary" size="small" @click="showCreateTenant">+ 创建租户</a-button>
          </div>
          <a-table
            :columns="tenantCols"
            :data-source="tenants"
            :loading="tenantLoading"
            row-key="tenant_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'status'">
                <a-badge :status="record.is_active ? 'success' : 'error'" />
                {{ record.is_active ? '正常' : '已禁用' }}
              </template>
              <template v-else-if="column.key === 'actions'">
                <a-space>
                  <a-button size="small" type="link" @click="resetKey(record)">重置Key</a-button>
                  <a-popconfirm
                    title="确认禁用该租户?"
                    @confirm="toggleTenant(record)"
                  >
                    <a-button size="small" type="link" :danger="record.is_active">
                      {{ record.is_active ? '禁用' : '启用' }}
                    </a-button>
                  </a-popconfirm>
                </a-space>
              </template>
            </template>
          </a-table>
        </div>
      </a-tab-pane>

      <!-- 系统状态 -->
      <a-tab-pane key="status" tab="系统状态">
        <div class="section">
          <div class="section-title">服务状态</div>
          <div class="status-grid">
            <div class="status-item">
              <div class="status-label">API服务</div>
              <div class="status-value">
                <a-badge status="success" />
                <span>运行中</span>
              </div>
              <div class="status-sub">P95: {{ sysStats.apiLatency }}ms</div>
            </div>
            <div class="status-item">
              <div class="status-label">向量库</div>
              <div class="status-value">
                <a-badge status="success" />
                <span>正常</span>
              </div>
              <div class="status-sub">{{ sysStats.vectorCount }} 条向量</div>
            </div>
            <div class="status-item">
              <div class="status-label">数据库</div>
              <div class="status-value">
                <a-badge status="success" />
                <span>已连接</span>
              </div>
              <div class="status-sub">{{ sysStats.dbSize }}</div>
            </div>
            <div class="status-item">
              <div class="status-label">内存占用</div>
              <div class="status-value">
                <a-badge status="success" />
                <span>{{ sysStats.memory }}MB</span>
              </div>
              <div class="status-sub">P95: {{ sysStats.qps }} QPS</div>
            </div>
          </div>
        </div>

        <div class="section" style="margin-top: 16px">
          <div class="section-title">实时QPS</div>
          <div class="chart-placeholder">
            <div class="chart-bars">
              <div
                v-for="(v, i) in qpsHistory"
                :key="i"
                class="chart-bar"
                :style="{ height: (v / maxQps * 100) + '%' }"
              >
                <span class="bar-label">{{ v }}</span>
              </div>
            </div>
            <div class="chart-x-axis">
              <span v-for="n in 12" :key="n">{{ n }}s</span>
            </div>
          </div>
        </div>
      </a-tab-pane>

      <!-- 模型配置 -->
      <a-tab-pane key="model" tab="模型配置">
        <div class="section">
          <div class="section-title">LLM 配置</div>
          <div class="form-grid">
            <div class="form-item">
              <div class="form-label">接口地址</div>
              <a-input
                v-model:value="modelConfig.baseUrl"
                placeholder="http://localhost:11434/v1"
                size="large"
              />
            </div>
            <div class="form-item">
              <div class="form-label">API密钥</div>
              <a-input-password
                v-model:value="modelConfig.apiKey"
                placeholder="sk-xxx"
                size="large"
              />
            </div>
            <div class="form-item">
              <div class="form-label">模型名称</div>
              <a-input
                v-model:value="modelConfig.model"
                placeholder="qwen2.5:7b"
                size="large"
              />
            </div>
            <div class="form-item">
              <div class="form-label">上下文窗口</div>
              <a-input-number
                v-model:value="modelConfig.maxTokens"
                :min="1000"
                :max="200000"
                :step="1000"
                size="large"
                style="width: 100%"
              />
            </div>
          </div>
          <div style="margin-top: 16px; display: flex; gap: 8px">
            <a-button type="primary" @click="testConnection">连接测试</a-button>
            <a-button @click="saveConfig">保存配置</a-button>
          </div>
          <div v-if="testResult" class="test-result" :class="testResult.ok ? 'success' : 'error'">
            {{ testResult.msg }}
          </div>
        </div>
      </a-tab-pane>

      <!-- 日志查看 -->
      <a-tab-pane key="logs" tab="日志查看">
        <div class="section">
          <div class="section-header">
            <span class="section-title">实时日志</span>
            <a-space>
              <a-select v-model:value="logLevel" style="width: 100px">
                <a-select-option value="info">INFO</a-select-option>
                <a-select-option value="warn">WARN</a-select-option>
                <a-select-option value="error">ERROR</a-select-option>
              </a-select>
              <a-button size="small" @click="loadLogs">刷新</a-button>
            </a-space>
          </div>
          <div class="log-viewer">
            <div
              v-for="(log, i) in logs"
              :key="i"
              :class="['log-line', log.level]"
            >
              <span class="log-time">{{ log.time }}</span>
              <span :class="['log-level', log.level]">{{ log.level.toUpperCase() }}</span>
              <span class="log-msg">{{ log.msg }}</span>
            </div>
            <div v-if="logs.length === 0" class="log-empty">暂无日志</div>
          </div>
        </div>
      </a-tab-pane>
    </a-tabs>

    <!-- 创建租户弹窗 -->
    <a-modal v-model:open="createVisible" title="创建租户" @ok="doCreateTenant">
      <div class="form-item">
        <div class="form-label">租户名称</div>
        <a-input v-model:value="newTenant.name" placeholder="公司名称" />
      </div>
      <div class="form-item" style="margin-top: 12px">
        <div class="form-label">套餐</div>
        <a-select v-model:value="newTenant.plan" style="width: 100%">
          <a-select-option value="standard">标准版</a-select-option>
          <a-select-option value="enterprise">企业版</a-select-option>
        </a-select>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { message } from 'ant-design-vue'

const activeTab = ref('tenants')

// 租户管理
const tenants = ref([])
const tenantLoading = ref(false)
const createVisible = ref(false)
const newTenant = reactive({ name: '', plan: 'standard' })

const tenantCols = [
  { title: '租户ID', key: 'tenant_id', dataIndex: 'tenant_id', width: 200, ellipsis: true },
  { title: '名称', key: 'name', dataIndex: 'name' },
  { title: '套餐', key: 'plan', dataIndex: 'plan', width: 100 },
  { title: '创建时间', key: 'created_at', dataIndex: 'created_at', width: 180 },
  { title: '状态', key: 'status', width: 100 },
  { title: '操作', key: 'actions', width: 160 }
]

async function loadTenants() {
  tenantLoading.value = true
  // 模拟数据
  tenants.value = [
    {
      tenant_id: 'tnt_demo_001',
      name: '演示租户',
      plan: 'standard',
      is_active: true,
      created_at: '2026-04-01 10:00:00'
    }
  ]
  tenantLoading.value = false
}

function showCreateTenant() {
  Object.assign(newTenant, { name: '', plan: 'standard' })
  createVisible.value = true
}

async function doCreateTenant() {
  if (!newTenant.name) {
    message.warning('请输入租户名称')
    return
  }
  // 模拟创建
  tenants.value.push({
    tenant_id: 'tnt_' + Date.now(),
    name: newTenant.name,
    plan: newTenant.plan,
    is_active: true,
    created_at: new Date().toLocaleString()
  })
  createVisible.value = false
  message.success('创建成功')
}

function resetKey(record) {
  const newKey = 'sk_' + Math.random().toString(36).substr(2)
  record.api_key = newKey
  message.success('API Key已重置')
}

function toggleTenant(record) {
  record.is_active = !record.is_active
  message.success(record.is_active ? '已启用' : '已禁用')
}

// 系统状态
const sysStats = ref({
  apiLatency: '45',
  vectorCount: '12,580',
  dbSize: '256 MB',
  memory: '1.2GB',
  qps: '128'
})

const qpsHistory = ref([45, 62, 38, 75, 88, 52, 93, 67, 41, 78, 55, 69])
const maxQps = 100

// 模型配置
const modelConfig = reactive({
  baseUrl: 'http://localhost:11434/v1',
  apiKey: '',
  model: 'qwen2.5:7b',
  maxTokens: 8192
})
const testResult = ref(null)

function testConnection() {
  testResult.value = { ok: true, msg: '连接成功，模型响应正常' }
  setTimeout(() => { testResult.value = null }, 5000)
}

function saveConfig() {
  message.success('配置已保存')
}

// 日志
const logLevel = ref('info')
const logs = ref([
  { time: '14:32:01', level: 'info', msg: 'GET /v1/memory/query 200 45ms' },
  { time: '14:32:05', level: 'info', msg: 'POST /v1/memory/add 200 23ms' },
  { time: '14:32:10', level: 'warn', msg: '检索延迟略高: 180ms' },
  { time: '14:32:15', level: 'error', msg: '连接超时 retry 1/3' },
  { time: '14:32:20', level: 'info', msg: '记忆合并完成 3->1 条' },
])

function loadLogs() {
  // 模拟刷新
}

onMounted(() => {
  loadTenants()
})
</script>

<style scoped>
.admin-page {
  min-height: calc(100vh - 120px);
}
.section {
  background: #1e1e1e;
  border-radius: 8px;
  padding: 16px;
}
.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.section-title {
  font-size: 14px;
  font-weight: 600;
  color: #e0e0e0;
  margin-bottom: 12px;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.status-item {
  background: #141414;
  border-radius: 6px;
  padding: 14px;
}
.status-label {
  font-size: 12px;
  color: #555;
  margin-bottom: 6px;
}
.status-value {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 16px;
  font-weight: 600;
  color: #e0e0e0;
}
.status-sub {
  font-size: 11px;
  color: #555;
  margin-top: 4px;
}
.chart-placeholder {
  background: #141414;
  border-radius: 6px;
  padding: 16px;
}
.chart-bars {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 80px;
}
.chart-bar {
  flex: 1;
  background: #6366f1;
  border-radius: 3px 3px 0 0;
  min-height: 4px;
  position: relative;
}
.bar-label {
  position: absolute;
  top: -18px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 10px;
  color: #555;
}
.chart-x-axis {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  font-size: 10px;
  color: #444;
}
.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.form-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.form-label {
  font-size: 12px;
  color: #666;
}
.log-viewer {
  background: #0d0d0d;
  border-radius: 6px;
  padding: 12px;
  max-height: 300px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  font-size: 12px;
}
.log-line {
  display: flex;
  gap: 8px;
  margin-bottom: 4px;
}
.log-time { color: #555; }
.log-level { font-weight: 600; min-width: 50px; }
.log-level.info { color: #22c55e; }
.log-level.warn { color: #eab308; }
.log-level.error { color: #ef4444; }
.log-msg { color: #aaa; }
.log-empty {
  text-align: center;
  color: #444;
  padding: 40px;
}
.test-result {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
}
.test-result.success { background: #22c55e20; color: #22c55e; }
.test-result.error { background: #ef444420; color: #ef4444; }
</style>
