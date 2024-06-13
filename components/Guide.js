import Link from "next/link";
import Image from "next/image";
import { guides } from "../lib/data";
import { GoArrowRight } from "react-icons/go";

const Guide = () => {
  return (
    <section className="flex flex-col items-center w-[80%]">
      <h2 className="mb-1">Guides</h2>
      <p className="text-center mb-6">
        These resources helped me grasp the theory of ML, the main use-cases in
        MLOps and how to contribute to open source projects like scikit-learn. I
        use these extensively for teaching.
      </p>

      <div className="w-full flex flex-col lg:flex-row justify-center gap-5">
        {guides.map((l) => (
          <div className="bg-black w-full rounded-[15px] lg:w-[33%]">
            <Link
              key={l.title}
              href={l.url}
              className="card bg-background border border-black h-full"
            >
              <div className="flex flex-col">
                <Image className="mb-6" alt={l.title} src={l.img} width={300} height={100} />
                
                <div className="mb-6">
                  <h3 className="mb-1">{l.title}</h3>
                  <p>{l.desc}</p>
                </div>
              </div>

              <button className="button flex justify-between gap-3 items-center bg-[#a78ce4]">
                Read <GoArrowRight />
              </button>
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Guide;
