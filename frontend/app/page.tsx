import DraftBoard from "@/components/DraftBoard";
import SiteFooter from "@/components/SiteFooter";

// min-h-screen lives here rather than on the board: the footer is a sibling of DraftBoard (so it
// stays server-rendered), and if the board kept the full-height rule the footer would be pushed
// a whole viewport below the content on tall displays.
export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1">
        <DraftBoard />
      </div>
      <SiteFooter blurb="Recommendations fuse a trained win-prob model with empirical map stats" />
    </div>
  );
}
