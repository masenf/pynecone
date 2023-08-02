/* _csr: Client Side Router entry point
 *
 * Use next/router to redirect to dynamic routes on the client side
 * when serving a static export.
 * 
 * This is intended for use with a `try_files` directive as a last fallback if
 * the requested URI is not found for file-based routing.
 * 
 * nginx: try_files $uri $uri.html /_csr.html
 * Caddy: try_files {path} {path}.html /_csr.html
 */
import Router from "next/router";
import { useEffect, useState } from "react";

export default function ClientSideRouting() {
  useEffect(() => {
    const doReplace = async () => {
      await Router.replace(window.location.pathname);
    }
    doReplace().catch((e) => {
      // on router error: stealth replace with 404 page, without modifying the requested URL
      Router.replace("/404", window.location.pathname);
    })
  }, []);

  return null;
}
