import Head from "next/head";
import styles from "../styles/Home.module.css";
import Layout from "../components/Layout";
import Hero from "../components/Hero";
import Guide from "../components/Guide";
import Teaching from "../components/Teaching";
import Notebook from "../components/Notebook";
import Conference from "../components/Conference";

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Vincent Maladiere</title>
        <link rel="icon" href="/favicon.ico" />

      </Head>
      <main className="bg-background w-[100vw] min-h-[100vh] overflow-hidden">
        <Layout>
          <Hero />
          <Guide />
          <Teaching />
          <Conference />
          <Notebook />
        </Layout>
      </main>
    </div>
  );
}
