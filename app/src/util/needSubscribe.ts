// [patched] 关闭订阅提示：始终视为已订阅/已付费。
// 保留 tip 形参以维护调用方签名，但不设默认值，
// 避免 window.siyuan.languages 尚未就绪时求值抛异常。
export const needSubscribe = (_tip?: string) => {
    return false;
};

/**
 * 判断是否可以使用第三方同步
 */
export const isPaidUser = () => {
    return true;
};
