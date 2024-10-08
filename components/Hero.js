import Link from "next/link";
import Image from "next/image";
import { socials, libs } from "../lib/data";

const Hero = () => {
  return (
    <section className="flex flex-col justify-between gap-10">
      <div className="flex flex-col 2xl:flex-row item-start photo-card">
        <Image
          alt="profile picture of Vincent Maladiere"
          src="/circle_head.png"
          width={300}
          height={300}
          className="mb-6 2xl:mr-10"
        />

        <div className="flex flex-col shrink-[5]">
          <h2 className="mb-1">Hey there!</h2>
          <p className="mb-6">
          I’m Vincent Maladière, an enthusiast of computer science, applied mathematics,
          and data science.
          </p>
          <p className="mb-6"> 
          I have developed open-source software and deployed machine learning models in
          healthcare and fintech companies, and co-authored a paper on survival
          analysis. I have also taught data science at École Polytechnique and Dauphine.
          </p>
          <p className="mb-6"> 
            I work at Probabl as a co-founding ML engineer, where I develop the skrub
            and hazardous libraries. You can reach me at:
          </p>
          
          <div className="flex flex-wrap flex-row justify-start gap-2">
            {socials.map((l) => (
              <Link key={l.title} href={l.url} className="social-button">
                {l.icon} {l.title}
              </Link>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <h3 className="text-center mb-1">I contributed to</h3>
        <div className="w-full flex-wrap md:flex-nowrap flex justify-center items-center align-center gap-5 md:gap-[20%]">
          {libs.map((l) => (
            <Link key={l.title} href={l.url}>
              <Image alt={l.title} src={l.img} width={150} height={150} />
            </Link>
          ))}
        </div>
      </div>

      <div className="flex gap-5">
        <div className="paper-callout">
          <h3 className="mb-1">I took part in this paper</h3>
          <Link
            key="competing risks"
            href="https://arxiv.org/abs/2406.14085"
          >
            <p className="underline">
              Teaching Models To Survive: Proper Scoring Rule and Stochastic
              Optimization with Competing Risks
            </p>
          </Link>
          <p>published in arxiv (2024)</p>
          <p>
            We propose a new SOTA model for survival analysis and competing risk
            analysis, trained on a proper scoring rule and plugged into a
            stochastic boosting tree. It is also much faster to train than
            current alternatives.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Hero;
