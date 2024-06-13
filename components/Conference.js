import Link from "next/link";
import Image from "next/image";
import { conferences } from "../lib/data";
import { GoArrowRight } from "react-icons/go";

const Conference = () => {
  return (
    <section className="flex flex-col items-center w-[80%]">
      <h2 className="text-center mb-1">Conferences</h2>
      <p className="text-center subtitle mb-6">
        I talked at the following events and conferences
      </p>

      <div className="flex flex-col justify-center gap-5 w-full md:w-[80%]">
        {conferences.map((l) => (
          <div className="bg-black w-full rounded-[15px]">
            <Link
              key={l.title}
              href={l.url}
              className="conf-card bg-background flex-col md:flex-row items-start md:items-center"
            >
              <div className="flex flex-col md:flex-row md:items-center gap-3 items-start">
                <Image alt={l.title} src={l.img} width={48} height={48} />

                <div>
                  <p className="font-bold mb-1">{l.title}</p>
                  <p>
                    {l.venue}, {l.city}
                  </p>
                </div>
              </div>

              <button className="button bg-[#a78ce4]">
                <GoArrowRight />
              </button>
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Conference;
