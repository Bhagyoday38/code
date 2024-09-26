import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Scanner;

public class WebScraper {

    public static void main(String[] args) {
        try {
            
            URL url = new URL("https://www.nytimes.com/");

            
            Scanner scanner = new Scanner(url.openStream());
            String htmlContent = "";
            while (scanner.hasNext()) {
                htmlContent += scanner.nextLine();
            }

            
            Document doc = Jsoup.parse(htmlContent);

           
            String title = doc.title();

            
            Elements links = doc.select("a");

            
            System.out.println("Title: " + title);
            System.out.println("Links:");
            for (Element link : links) {
                System.out.println(link.text() + " - " + link.attr("href"));
            }

        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
