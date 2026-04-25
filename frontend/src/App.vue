<template>
  <a-config-provider :theme="darkTheme ? darkThemeConfig : lightThemeConfig">
    <a-layout class="app-layout">
      <a-layout-sider
        v-model:collapsed="collapsed"
        :trigger="null"
        collapsible
        :style="{ background: '#141414' }"
      >
        <div class="logo">
          <span class="logo-text">{{ darkTheme ? 'PMC' : 'PMC' }}</span>
          <span v-if="!collapsed" class="logo-sub">Private Memory Core</span>
        </div>
        <a-menu
          v-model:selectedKeys="currentKeys"
          theme="dark"
          mode="inline"
          @click="handleMenu"
        >
          <a-menu-item key="console">
            <ConsoleOutlined />
            <span>控制台</span>
          </a-menu-item>
          <a-menu-item key="memory">
            <DatabaseOutlined />
            <span>记忆管理</span>
          </a-menu-item>
          <a-menu-item key="admin">
            <SettingOutlined />
            <span>系统管理</span>
          </a-menu-item>
        </a-menu>
      </a-layout-sider>

      <a-layout>
        <a-layout-header class="header">
          <div class="header-left">
            <span class="page-title">{{ pageTitle }}</span>
          </div>
          <div class="header-right">
            <a-tag :color="envColor">{{ envLabel }}</a-tag>
            <a-switch
              v-model:checked="darkTheme"
              checked-children="暗"
              un-checked-children="明"
              style="margin: 0 12px"
            />
            <a-dropdown>
              <a-avatar class="avatar">{{ userInitials }}</a-avatar>
              <template #overlay>
                <a-menu>
                  <a-menu-item key="info">
                    <UserOutlined /> 租户信息
                  </a-menu-item>
                  <a-menu-divider />
                  <a-menu-item key="logout" danger>
                    <LogoutOutlined /> 退出登录
                  </a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>
          </div>
        </a-layout-header>

        <a-layout-content class="content">
          <router-view />
        </a-layout-content>
      </a-layout>
    </a-layout>
  </a-config-provider>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  ConsoleOutlined, DatabaseOutlined, SettingOutlined,
  UserOutlined, LogoutOutlined
} from '@ant-design/icons-vue'
import { theme } from 'ant-design-vue'

const router = useRouter()
const route = useRoute()
const collapsed = ref(false)
const darkTheme = ref(true)
const currentKeys = ref(['console'])

const pageTitles = {
  console: '控制台 · 演示中心',
  memory: '记忆管理中心',
  admin: '系统管理后台'
}
const pageTitle = computed(() => pageTitles[currentKeys.value[0]] || '控制台')

const envLabel = ref('本地部署')
const envColor = computed(() => envLabel.value === '本地部署' ? 'cyan' : 'green')

const darkThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: { colorPrimary: '#6366f1' }
}
const lightThemeConfig = {
  algorithm: theme.defaultAlgorithm,
  token: { colorPrimary: '#6366f1' }
}

const userInitials = computed(() => {
  return 'PM'
})

function handleMenu({ key }) {
  currentKeys.value = [key]
  router.push({ name: key })
}

watch(() => route.name, (name) => {
  if (name) currentKeys.value = [name]
}, { immediate: true })
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
}
.logo {
  height: 56px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 8px;
  border-bottom: 1px solid #2a2a2a;
}
.logo-text {
  font-size: 18px;
  font-weight: 700;
  color: #6366f1;
  letter-spacing: 2px;
}
.logo-sub {
  font-size: 11px;
  color: #666;
}
.header {
  background: #1a1a1a;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  border-bottom: 1px solid #2a2a2a;
}
.header-left .page-title {
  font-size: 15px;
  font-weight: 500;
  color: #e0e0e0;
}
.header-right {
  display: flex;
  align-items: center;
}
.avatar {
  background: #6366f1;
  cursor: pointer;
}
.content {
  margin: 0;
  padding: 20px;
  background: #0d0d0d;
  min-height: calc(100vh - 64px);
}
</style>
