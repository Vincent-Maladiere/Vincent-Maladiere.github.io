const nextConfig = {
  output: "export",  // <=== enables static exports
  reactStrictMode: true,
  distDir: 'build',
  images: {
    "unoptimized": true,
  },
}
const withNextra = require("nextra")({
  theme: "nextra-theme-docs",
  themeConfig: "./theme.config.jsx",
  latex: true,
});

module.exports = withNextra(nextConfig);

// If you have other Next.js configurations, you can pass them as the parameter:
// module.exports = withNextra({ /* other next.js config */ })
