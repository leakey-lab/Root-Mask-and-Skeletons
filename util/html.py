import dominate
from dominate.tags import *
import os


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

    It consists of functions such as <add_header> (add a text header to the HTML file),
    <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
    It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        with self.doc.head:
            meta(charset="utf-8")
            meta(name="viewport", content="width=device-width, initial-scale=1.0")
            link(
                rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
            )
            style(
                """
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f4f4f4; }
                .container { width: 90%; max-width: 1200px; margin: auto; padding: 20px; }
                h1, h3 { color: #2c3e50; text-align: center; }
                .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; justify-items: center; }
                .image-item { background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 0 15px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
                .image-item img { width: 100%; height: 300px; object-fit: cover; border-radius: 5px; }
                .image-item p { margin-top: 10px; text-align: center; font-size: 14px; word-wrap: break-word; }
                .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.9); }
                .modal-content { margin: auto; display: block; max-width: 80%; max-height: 80%; }
                .modal-nav { position: absolute; top: 50%; transform: translateY(-50%); color: white; font-size: 40px; cursor: pointer; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; }
                .close { color: #f1f1f1; position: absolute; top: 15px; right: 35px; font-size: 40px; font-weight: bold; cursor: pointer; }
                #prevButton { left: 20px; }
                #nextButton { right: 20px; }
                @media (max-width: 768px) {
                    .image-grid { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
                    .image-item { max-width: 100%; }
                }
            """
            )
            script(
                """
                function openModal(imgElement) {
                    var modal = document.getElementById("imageModal");
                    var modalImg = document.getElementById("modalImage");
                    modal.style.display = "block";
                    modalImg.src = imgElement.src;
                    currentIndex = Array.from(document.getElementsByClassName("grid-image")).indexOf(imgElement);
                }
                function closeModal() {
                    document.getElementById("imageModal").style.display = "none";
                }
                function changeImage(n) {
                    var images = document.getElementsByClassName("grid-image");
                    currentIndex = (currentIndex + n + images.length) % images.length;
                    document.getElementById("modalImage").src = images[currentIndex].src;
                }
                var currentIndex = 0;
            """
            )
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            with div(cls="container"):
                h1(text)

    def add_images(self, ims, txts, links, width=300):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = div(cls="image-grid")
        self.doc.add(self.t)
        for im, txt, link in zip(ims, txts, links):
            with self.t:
                with div(cls="image-item"):
                    with a(
                        href="javascript:void(0);",
                        onclick=f"openModal(this.getElementsByTagName('img')[0])",
                    ):
                        img(cls="grid-image", src=os.path.join("images", im))
                    p(txt)

        # Add modal for full-size image view
        with self.doc:
            with div(id="imageModal", cls="modal"):
                span("×", cls="close", onclick="closeModal()")
                img(cls="modal-content", id="modalImage")
                span("❮", cls="modal-nav", id="prevButton", onclick="changeImage(-1)")
                span("❯", cls="modal-nav", id="nextButton", onclick="changeImage(1)")

    def save(self):
        """save the current content to the HTML file"""
        html_file = "%s/index.html" % self.web_dir
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(self.doc.render())


if __name__ == "__main__":  # we show an example usage here.
    html = HTML("web/", "test_html")
    html.add_header("Image Gallery")

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append("image_%d.png" % n)
        txts.append("text_%d" % n)
        links.append("image_%d.png" % n)
    html.add_images(ims, txts, links)
    html.save()
