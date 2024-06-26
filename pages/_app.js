import "../styles/global.css";
import localfont from "next/font/local";
import { GoogleAnalytics } from '@next/third-parties/google'


const polySans = localfont({
  src: [
    {
      path: "../public/fonts/PolySansMedian.ttf",
      weight: "600",
      style: "bold",
    },
    {
      path: "../public/fonts/PolySansNeutral.ttf",
      weight: "400",
      style: "normal",
    },
    {
      path: "../public/fonts/PolySansSlim.ttf",
      weight: "200",
      style: "thin",
    },
  ],
});

const App = ({ Component, pageProps }) => {
  return (
    <main className={polySans.className}>
      <Component {...pageProps} />
      <GoogleAnalytics gaId="G-NQ78ZVKXEN" />
    </main>
  );
};

export default App;
