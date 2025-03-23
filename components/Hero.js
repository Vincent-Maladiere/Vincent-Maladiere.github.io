import Link from "next/link";
import Image from "next/image";
import { socials, libs } from "../lib/data";

const Hero = () => {
  return (
    <section className="flex flex-col justify-between gap-10">
      <div className="flex flex-col photo-card">
      <div className="flex flex-col lg:flex-row item-start gap-10">
        <Image
          alt="profile picture of Vincent Maladiere"
          src="/circle_head.png"
          width={300}
          height={300}
          className="mb-6" 
        />

        <div className="flex flex-col ">
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
            I work at Probabl as a co-founding research engineer, where I develop the
            skrub and hazardous libraries. You can reach me at:
          </p>
        
        </div>
        </div>
          <div className="flex flex-wrap flex-row justify-start gap-2">
            {socials.map((l) => (
              <Link key={l.title} href={l.url} className="social-button">
                {l.icon} {l.title}
              </Link>
            ))}
          </div>
      </div>

      <div className="flex flex-col gap-3">
        <h3 className="text-center mb-1">I contributed to</h3>
        <div className="w-full flex-wrap md:flex-nowrap flex justify-center items-center align-center gap-5 md:gap-[10%]">
          {libs.map((l) => (
            <Link key={l.title} href={l.url}>
              <Image alt={l.title} src={l.img} width={150} height={150} />
            </Link>
          ))}
        </div>
      </div>

      <div className="flex gap-5">
        <div className="paper-callout">
          <h3 className="mb-1">I co-authored a research paper published in AISTATS 2025</h3>
          <Link
            key="competing risks"
            href="https://hal.science/hal-04617672"
          >
            <p className="underline">
            Survival Models: Proper Scoring Rule and Stochastic Optimization with Competing Risks
            </p>
          </Link>
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
