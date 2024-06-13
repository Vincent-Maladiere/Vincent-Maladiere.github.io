import { GoogleAnalytics } from '@next/third-parties/google'

export default function RootLayout({ children }) {
    return (
      <html lang="en">
        <body>
          {/* Layout UI */}
          <main>{children}</main>
        </body>
        <GoogleAnalytics gaId="GTM-MXFNFJ4R" />
      </html>
    )
  }