<template>
  <div class="console-page">
    <!-- 顶部：用户/统计 -->
    <div class="console-top">
      <div class="stat-card">
        <div class="stat-label">记忆总数</div>
        <div class="stat-value blue">{{ stats.total }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">压缩率</div>
        <div class="stat-value green">{{ stats.compression }}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">对话轮次</div>
        <div class="stat-value purple">{{ chatCount }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">响应耗时</div>
        <div class="stat-value orange">{{ lastLatency }}ms</div>
      </div>
    </div>

    <!-- 主内容 -->
    <div class="console-main">
      <!-- 左侧：记忆面板 -->
      <div class="memory-panel">
        <!-- 记忆tabs -->
        <div class="panel-section">
          <div class="section-title">记忆分类</div>
          <a-tabs v-model:activeKey="activeTab" size="small">
            <a-tab-pane key="all" tab="全部" />
            <a-tab-pane key="fact" tab="事实" />
            <a-tab-pane key="preference" tab="偏好" />
            <a-tab-pane key="belief" tab="信念" />
          </a-tabs>
        </div>

        <!-- 搜索框 -->
        <div class="panel-section">
          <a-input-search
            v-model:value="searchQuery"
            placeholder="搜索记忆..."
            allow-clear
            @search="doSearch"
          />
        </div>

        <!-- 记忆列表 -->
        <div class="memory-list">
          <div v-if="loadingMemories" class="loading-wrap">
            <a-spin />
          </div>
          <div v-else-if="displayMemories.length === 0" class="empty">
            暂无记忆
          </div>
          <div
            v-for="mem in displayMemories"
            :key="mem.id"
            class="memory-item"
            @click="selectMemory(mem)"
          >
            <div class="mem-content">{{ mem.content.substring(0, 80) }}</div>
            <div class="mem-meta">
              <a-tag :color="typeColor(mem.metadata?.type)" size="small">
                {{ mem.metadata?.type || 'fact' }}
              </a-tag>
              <span class="mem-time">{{ formatTime(mem.timestamp) }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧：聊天窗口 -->
      <div class="chat-panel">
        <!-- 对话历史 -->
        <div class="chat-history" ref="chatRef">
          <div v-if="messages.length === 0" class="chat-empty">
            <div class="chat-empty-icon">💬</div>
            <div>开始对话，体验记忆能力</div>
          </div>
          <div
            v-for="(msg, i) in messages"
            :key="i"
            :class="['chat-bubble', msg.role]"
          >
            <div class="bubble-content">{{ msg.content }}</div>
            <div class="bubble-time">{{ msg.time }}</div>
          </div>
          <div v-if="chatLoading" class="chat-bubble assistant">
            <a-spin size="small" />
          </div>
        </div>

        <!-- 输入区 -->
        <div class="chat-input-area">
          <a-input-group compact class="user-id-row">
            <a-input
              v-model:value="chatUserId"
              placeholder="用户ID"
              style="width: 140px"
            />
          </a-input-group>
          <div class="input-row">
            <a-input
              v-model:value="chatInput"
              placeholder="输入问题，按 Enter 发送..."
              @pressEnter="sendMessage"
              :disabled="chatLoading"
            />
            <a-button
              type="primary"
              @click="sendMessage"
              :loading="chatLoading"
            >
              发送
            </a-button>
          </div>
        </div>

        <!-- 底部信息栏 -->
        <div class="chat-footer">
          <span>响应: {{ lastLatency }}ms</span>
          <span>召回: {{ lastRecallCount }}条</span>
          <a-button type="text" size="small" @click="clearChat">清空</a-button>
          <a-button type="text" size="small" @click="exportLog">导出日志</a-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import { addMemory, queryMemory, chat } from '@/api'

const stats = ref({ total: 0, compression: '89' })
const chatCount = ref(0)
const lastLatency = ref(0)
const lastRecallCount = ref(0)
const loadingMemories = ref(false)
const chatLoading = ref(false)

const activeTab = ref('all')
const searchQuery = ref('')
const displayMemories = ref([])
const allMemories = ref([])

const chatUserId = ref('user_demo')
const chatInput = ref('')
const messages = ref([])
const chatRef = ref(null)

const typeColor = (type) => {
  const map = { fact: 'blue', preference: 'pink', belief: 'purple', event: 'orange' }
  return map[type] || 'blue'
}

const formatTime = (ts) => {
  if (!ts) return ''
  return dayjs(ts * 1000).format('MM-DD HH:mm')
}

async function loadMemories() {
  loadingMemories.value = true
  try {
    const r = await queryMemory({ user_id: chatUserId.value, query: searchQuery.value || '', limit: 50 })
    displayMemories.value = r.memories || []
    allMemories.value = displayMemories.value
    stats.value.total = displayMemories.value.length
    messages.value = messages.value
  } catch (e) {
    console.error(e)
  } finally {
    loadingMemories.value = false
  }
}

function selectMemory(mem) {
  chatInput.value = mem.content
}

async function doSearch() {
  await loadMemories()
}

async function sendMessage() {
  const text = chatInput.value.trim()
  if (!text) return
  chatLoading.value = true

  const userMsg = { role: 'user', content: text, time: dayjs().format('HH:mm:ss') }
  messages.value.push(userMsg)
  chatInput.value = ''
  chatCount.value++
  scrollBottom()

  const start = Date.now()
  try {
    // 先自动存储记忆
    await addMemory({ user_id: chatUserId.value, content: text, metadata: { type: 'fact' } })

    const r = await chat({
      user_id: chatUserId.value,
      model: 'qwen',
      messages: messages.value.map(m => ({ role: m.role, content: m.content }))
    })

    lastLatency.value = Date.now() - start
    lastRecallCount.value = r.usage?.total_tokens || 0

    const reply = r.choices?.[0]?.message?.content || '（无响应）'
    messages.value.push({ role: 'assistant', content: reply, time: dayjs().format('HH:mm:ss') })

    await loadMemories()
  } catch (e) {
    messages.value.push({ role: 'assistant', content: '错误：' + e.message, time: dayjs().format('HH:mm:ss') })
  } finally {
    chatLoading.value = false
    scrollBottom()
  }
}

function scrollBottom() {
  nextTick(() => {
    if (chatRef.value) {
      chatRef.value.scrollTop = chatRef.value.scrollHeight
    }
  })
}

function clearChat() {
  messages.value = []
}

function exportLog() {
  const content = messages.value
    .map(m => `[${m.time}] ${m.role}: ${m.content}`)
    .join('\n')
  const blob = new Blob([content], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `su-memory-log-${dayjs().format('YYYYMMDD-HHmmss')}.txt`
  a.click()
}

onMounted(() => {
  loadMemories()
})
</script>

<style scoped>
.console-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 120px);
  gap: 16px;
}
.console-top {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  flex-shrink: 0;
}
.stat-card {
  background: #1e1e1e;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.stat-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 6px;
}
.stat-value {
  font-size: 24px;
  font-weight: 700;
}
.stat-value.blue { color: #6366f1; }
.stat-value.green { color: #22c55e; }
.stat-value.purple { color: #a855f7; }
.stat-value.orange { color: #f97316; }
.console-main {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 16px;
  flex: 1;
  min-height: 0;
}
.memory-panel {
  background: #1e1e1e;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-section {
  padding: 12px 16px;
  border-bottom: 1px solid #2a2a2a;
}
.section-title {
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}
.memory-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.loading-wrap, .empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100px;
  color: #555;
  font-size: 13px;
}
.memory-item {
  padding: 10px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 4px;
  transition: background 0.2s;
}
.memory-item:hover {
  background: #2a2a2a;
}
.mem-content {
  font-size: 13px;
  color: #ccc;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mem-meta {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mem-time {
  font-size: 11px;
  color: #555;
}
.chat-panel {
  background: #1e1e1e;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.chat-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #444;
  gap: 8px;
}
.chat-empty-icon {
  font-size: 40px;
}
.chat-bubble {
  margin-bottom: 12px;
  max-width: 75%;
}
.chat-bubble.user {
  margin-left: auto;
  text-align: right;
}
.chat-bubble.user .bubble-content {
  background: #6366f1;
  color: #fff;
  border-radius: 12px 12px 0 12px;
  padding: 10px 14px;
  display: inline-block;
}
.chat-bubble.assistant .bubble-content {
  background: #2a2a2a;
  color: #ddd;
  border-radius: 12px 12px 12px 0;
  padding: 10px 14px;
  display: inline-block;
}
.bubble-content {
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
}
.bubble-time {
  font-size: 10px;
  color: #444;
  margin-top: 4px;
}
.chat-input-area {
  padding: 12px 16px;
  border-top: 1px solid #2a2a2a;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.user-id-row {
  margin-bottom: 0;
}
.input-row {
  display: flex;
  gap: 8px;
}
.input-row .ant-input {
  flex: 1;
}
.chat-footer {
  padding: 8px 16px;
  border-top: 1px solid #2a2a2a;
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 12px;
  color: #555;
  flex-shrink: 0;
}
</style>
