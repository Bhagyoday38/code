import java.util.Scanner;

public class FibonacciSequence {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter the number of terms: ");
        int numberOfTerms = scanner.nextInt();

        int firstTerm = 0;
        int secondTerm = 1;

        System.out.println(firstTerm);
        System.out.println(secondTerm);

        for (int i = 2; i < numberOfTerms; i++) {
            int nextTerm = firstTerm + secondTerm;
            System.out.println(nextTerm);
            firstTerm = secondTerm;
            secondTerm = nextTerm;
        }
    }
}
