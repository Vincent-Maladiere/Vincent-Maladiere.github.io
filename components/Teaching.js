import Link from "next/link";
import Image from "next/image";
import { courses } from "../lib/data";
import { GoArrowRight } from "react-icons/go";

const Teaching = () => {
  return (
    <section className="flex flex-col items-center w-[80%]">
      <h2 className="mb-1">Courses</h2>
      <p className="text-center subtitle mb-6">
        I teach data science at Polytechnique and Dauphine.
        <br />
        You can find below courses content and exercices.
      </p>
      <div className="w-full flex flex-col md:flex-row justify-center gap-5">
        {courses.map((l) => (
          <div className="bg-black w-full rounded-[15px] md:w-[45%]">
            <Link
              key={l.title}
              href={l.url}
              className="card bg-background border border-black h-full"
            >
              <Image className="mb-6" alt={l.title} src={l.img} width={170} height={170} />
              <div className="mb-6">
                <h3 className="mb-1">{l.title}</h3>
                <p className="mb-3">{l.school}</p>
                <p>{l.desc}</p>
              </div>
              
              <button className="button bg-[#a78ce4] flex justify-between gap-3 items-center">
                See course <GoArrowRight />
              </button>
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Teaching;
