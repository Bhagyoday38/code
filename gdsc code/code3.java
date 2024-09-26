import java.util.Scanner;

public class EvenIndexedCharacters {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();

        for (int i = 0; i < str.length(); i += 2) {
            System.out.print(str.charAt(i));
        }

        System.out.println();
    }
}
