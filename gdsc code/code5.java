import java.util.Scanner;

public class CheckFirstAndLast
 {
    public static void main(String[] args)
 {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the size of the list: ");
        int size = scanner.nextInt();
        System.out.println("Enter the elements of the list: ");
        int[] list = new int[size];
        for (int i = 0; i < size; i++) 
{
            list[i] = scanner.nextInt();
        }
        boolean isEqual = list[0] == list[size - 1];
        System.out.println("The first and last number of the list are the same: " + isEqual);
    }
}
